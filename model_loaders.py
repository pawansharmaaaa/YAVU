# This file is a part of https://github.com/pawansharmaaaa/Lip_Wise/ repository.

import file_check
import torch
import cv2
from facexlib.utils import load_file_from_url
from facexlib.parsing.bisenet import BiSeNet
from facexlib.parsing.parsenet import ParseNet

import numpy as np

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize
# from models import Wav2Lip
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

class model_loaders:

    def __init__(self, restorer, weight):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.weight = weight
        # self.wav2lip_model = self.load_wav2lip_model()
        if restorer == 'GFPGAN':
            self.restorer = self.load_gfpgan_model()
        elif restorer == 'CodeFormer':
            self.restorer = self.load_codeformer_model()

    def _load(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def load_realesrgan_model(self):

        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn("YAVU uses RealESRGAN for backgound upscaling and it's inference is really slow on CPU. Please consider using GPU.")
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            esr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            bg_upsampler = RealESRGANer(
                scale=netscale,
                model_path=file_check.REALESRGAN_MODEL_PATH,
                dni_weight=None,
                model=esr_model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
                gpu_id=None)
            
        return bg_upsampler

    def load_gfpgan_model(self):
        
        print(f"Load GFPGAN checkpoint from: {file_check.GFPGAN_MODEL_PATH}")
        gfpgan = GFPGANv1Clean(
                        out_size=512,
                        num_style_feat=512,
                        channel_multiplier=2,
                        decoder_load_path=None,
                        fix_decoder=False,
                        num_mlp=8,
                        input_is_latent=True,
                        different_w=True,
                        narrow=1,
                        sft_half=True)

        loadnet = torch.load(file_check.GFPGAN_MODEL_PATH)

        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'

        gfpgan.load_state_dict(loadnet[keyname], strict=True)
        restorer = gfpgan.eval()
        return restorer.to(self.device)

    def load_codeformer_model(self):
        print(f"Load CodeFormer checkpoint from: {file_check.CODEFORMERS_MODEL_PATH}")
        
        model = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to(self.device)

        ckpt_path = file_check.CODEFORMERS_MODEL_PATH
        checkpoint = torch.load(ckpt_path)['params_ema']
        model.load_state_dict(checkpoint)
        return model.eval()
    
    def restore_background(self, background, outscale=1.0):
        bgupsampler = self.load_realesrgan_model()
        if bgupsampler is not None:
            background = bgupsampler.enhance(background, outscale=outscale)
        return background
    
    def restore_wGFPGAN(self, dubbed_face):
        dubbed_face = cv2.resize(dubbed_face.astype(np.uint8) / 255., (512, 512), interpolation=cv2.INTER_LANCZOS4)
        dubbed_face_t = img2tensor(dubbed_face, bgr2rgb=True, float32=True)
        normalize(dubbed_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        dubbed_face_t = dubbed_face_t.unsqueeze(0).to(self.device)
        
        try:
            output = self.restorer(dubbed_face_t, return_rgb=False, weight=self.weight)[0]
            restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
        except RuntimeError as error:
            print(f'\tFailed inference for GFPGAN: {error}.')
            restored_face = tensor2img(dubbed_face_t.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
        
        restored_face = restored_face.astype(np.uint8)
        return restored_face
    
    def restore_wCodeFormer(self, dubbed_face):
        dubbed_face = cv2.resize(dubbed_face.astype(np.uint8) / 255., (512, 512), interpolation=cv2.INTER_LANCZOS4)
        dubbed_face_t = img2tensor(dubbed_face, bgr2rgb=True, float32=True)
        normalize(dubbed_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        dubbed_face_t = dubbed_face_t.unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                output = self.restorer(dubbed_face_t, w=self.weight, adain=True)[0]
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except RuntimeError as error:
            print(f'\tFailed inference for CodeFormer: {error}.')
            restored_face = tensor2img(dubbed_face_t, rgb2bgr=True, min_max=(-1, 1))
        
        restored_face = restored_face.astype(np.uint8)
        return restored_face