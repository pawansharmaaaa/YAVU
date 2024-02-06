import cv2
import os
import numpy as np

import helpers
import model_loaders
import file_check

def infer_image(image, restorer, model_name, weight, super_sample, outscale):
    
    # Create Objetcs
    fc = file_check.FileCheck(model_name)
    ml = model_loaders(restorer, model_name, weight)
    dim = helpers.FrameDimensions()
    det = helpers.ModelProcessors()
    help = helpers.FaceHelpers(image_mode=True, dimensions=dim)

    # Perform check again
    fc.file_check()

    # Load the image
    image_name = os.path.basename(image)
    original_img = cv2.imread(image)

    # Get The face Coordinates
    det.detect_for_image(original_img)

    # Preprocess the image
    mask, inv_mask, center, bbox = help.gen_face_mask()
    extracted_face = help.extract_face(original_img)
    cropped_face, aligned_bbox, rotmax = help.align_crop_face(extracted_face)

    # Feed to Model
    if restorer == 'GFPGAN':
        restored_face = ml.restore_wGFPGAN(cropped_face)
    elif restorer == 'CodeFormer':
        restored_face = ml.restore_wCodeFormer(cropped_face)

    # Postprocess the image
    processed_ready = help.paste_back_black_bg(restored_face, aligned_bbox, original_img)
    ready_to_paste = help.unwarp_align(processed_ready, rotmax)
    final_blend = help.paste_back(ready_to_paste, original_img, mask, inv_mask, center)

    if super_sample:
        final_blend = ml.restore_background(final_blend, outscale=outscale)

    save_path = os.path.join(fc.OUTPUT_DIR, image_name)

    cv2.imwrite(save_path, final_blend)

    return save_path

def infer_video(video, restorer, model_name, weight, super_sample, outscale):

    # Create Objetcs