import os
from basicsr.utils.download_util import load_file_from_url

class FileCheck:
    def __init__(self, real_esr_gan_model_name):

        self.real_esr_gan_model_name = real_esr_gan_model_name

        self.LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
        self.DETECTOR_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite'
        self.GFPGAN_MODEL_URL = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        self.CODEFORMERS_MODEL_URL = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
        self.REAL_ESRGAN_MODEL_URL = {
            'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'RealESRNet_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
            'RealESRGAN_x4plus_anime_6B': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
            'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            'realesr-animevideov3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
            'realesr-general-wdn-x4v3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'realesr-general-x4v3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        }

        self.CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

        self.RESULTS_DIR = os.path.join(self.CURRENT_FILE_DIRECTORY, 'results')
        self.WEIGHTS_DIR = os.path.join(self.CURRENT_FILE_DIRECTORY, 'weights')
        self.MP_WEIGHTS_DIR = os.path.join(self.WEIGHTS_DIR, 'mp')
        self.GFPGAN_WEIGHTS_DIR = os.path.join(self.WEIGHTS_DIR, 'gfpgan')
        self.CODEFORMERS_WEIGHTS_DIR = os.path.join(self.WEIGHTS_DIR, 'codeformers')
        self.REALESRGAN_WEIGHTS_DIR = os.path.join(self.WEIGHTS_DIR, 'realesrgan')

        self.TEMP_DIR = os.path.join(self.CURRENT_FILE_DIRECTORY, 'temp')
        self.NPY_FILES_DIR = os.path.join(self.TEMP_DIR, 'npy_files')
        self.MEDIA_DIR = os.path.join(self.TEMP_DIR, 'media')

        self.OUTPUT_DIR = os.path.join(self.CURRENT_FILE_DIRECTORY, 'output')

        self.MP_LANDMARKER_MODEL_PATH = os.path.join(self.MP_WEIGHTS_DIR, 'face_landmarker.task')
        self.MP_DETECTOR_MODEL_PATH = os.path.join(self.MP_WEIGHTS_DIR, 'blaze_face_short_range.tflite')
        self.GFPGAN_MODEL_PATH = os.path.join(self.GFPGAN_WEIGHTS_DIR, 'GFPGANv1.4.pth')
        self.CODEFORMERS_MODEL_PATH = os.path.join(self.CODEFORMERS_WEIGHTS_DIR, 'codeformer.pth')
        self.REALESRGAN_MODEL_PATH = os.path.join(self.REALESRGAN_WEIGHTS_DIR, f'{self.real_esr_gan_model_name}.pth')

        self.perform_check()

    def perform_check(self):
        try:
            if not os.path.exists(self.TEMP_DIR):
                os.makedirs(self.TEMP_DIR)
                os.makedirs(self.NPY_FILES_DIR)
                os.makedirs(self.MEDIA_DIR)

            if not os.path.exists(self.OUTPUT_DIR):
                os.makedirs(self.OUTPUT_DIR)

            if not os.path.exists(self.WEIGHTS_DIR):
                os.makedirs(self.WEIGHTS_DIR)
                os.makedirs(self.MP_WEIGHTS_DIR)
                os.makedirs(self.GFPGAN_WEIGHTS_DIR)
                os.makedirs(self.CODEFORMERS_WEIGHTS_DIR)
                os.makedirs(self.REALESRGAN_WEIGHTS_DIR)

            if not os.path.exists(self.MP_LANDMARKER_MODEL_PATH):
                print("Downloading Face Landmarker model...")
                load_file_from_url(url=self.LANDMARKER_MODEL_URL,
                                   model_dir=self.MP_WEIGHTS_DIR,
                                   progress=True,
                                   file_name='face_landmarker.task')

            if not os.path.exists(self.MP_DETECTOR_MODEL_PATH):
                print("Downloading Face Detector model...")
                load_file_from_url(url=self.DETECTOR_MODEL_URL,
                                   model_dir=self.MP_WEIGHTS_DIR,
                                   progress=True,
                                   file_name='blaze_face_short_range.tflite')

            if not os.path.exists(self.GFPGAN_MODEL_PATH):
                print("Downloading GFPGAN model...")
                load_file_from_url(url=self.GFPGAN_MODEL_URL,
                                   model_dir=self.GFPGAN_WEIGHTS_DIR,
                                   progress=True,
                                   file_name='GFPGANv1.4.pth')

            if not os.path.exists(self.CODEFORMERS_MODEL_PATH):
                print("Downloading CodeFormer model...")
                load_file_from_url(url=self.CODEFORMERS_MODEL_URL,
                                   model_dir=self.CODEFORMERS_WEIGHTS_DIR,
                                   progress=True,
                                   file_name='codeformer.pth')

            if not os.path.exists(self.REALESRGAN_MODEL_PATH):
                print(f"Downloading Real-ESRGAN model: {self.REALESRGAN_MODEL_PATH}...")
                load_file_from_url(url=self.REAL_ESRGAN_MODEL_URL[self.real_esr_gan_model_name],
                                   model_dir=self.REALESRGAN_WEIGHTS_DIR,
                                   progress=True,
                                   file_name=f'{self.real_esr_gan_model_name}.pth')

        except OSError as e:
            print(f"OS Error occurred: {e}")
        except Exception as e:
            print(f"Unexpected Error occurred while performing file_check: {e}")

    def get_file_type(self, filename):
        image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
        video_extensions = ["mp4", "mov", "avi", "mkv", "flv"]
        audio_extensions = ["mp3", "wav", "flac", "ogg", "m4a"]

        extension = filename.split('.')[-1].lower()

        if extension in image_extensions:
            return "image", extension
        elif extension in video_extensions:
            return "video", extension
        elif extension in audio_extensions:
            return "audio", extension
        else:
            return "unknown", extension

if __name__ == "__main__":
    file_check = FileCheck(real_esr_gan_model_name='RealESRGAN_x4plus')
    file_check.perform_check()
