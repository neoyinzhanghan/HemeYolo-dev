##############################################################################################
# PACKAGE IMPORT
##############################################################################################

import os
from PIL import Image
import glob
from tqdm import tqdm
import argparse



##############################################################################################
# DEFINE FUNCTIONS
##############################################################################################

def cut_image_into_quadrants(img_path):
    # Open image file
    img = Image.open(img_path)

    # Verify image dimensions ### no need as the code is adaptive to size
    if img.size[0] < 512 or img.size[1] < 512:
        raise ValueError(f"Image width and height must both be at least 512 pixels, your image is {img.size[0]}x{img.size[1]} pixels!")

    # Define box parameters (top-left, top-right, bottom-left, bottom-right)
    box_TL = (0, 0, 512, 512)
    box_TR = (img.size[0] - 512, 0, img.size[0], 512)
    box_BL = (0, img.size[1]-512, 512, img.size[1])
    box_BR = (img.size[0] - 512, img.size[1]-512, img.size[0], img.size[1])

    # Crop images and save
    img_TL = img.crop(box_TL)
    img_TR = img.crop(box_TR)
    img_BL = img.crop(box_BL)
    img_BR = img.crop(box_BR)

    # Return as tuple
    return (img_TL, img_TR, img_BL, img_BR)

def cut_images_in_folder(input_folder, output_folder):

    # Get list of jpg files in the input folder
    jpg_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    # Iterate through each .jpg file in the input folder with a progress bar
    for img_path in tqdm(jpg_files, desc="Processing images"):
        # Extract the base filename without extension
        base_filename = os.path.basename(os.path.splitext(img_path)[0])

        # Apply the previous function
        img_TL, img_TR, img_BL, img_BR = cut_image_into_quadrants(img_path)

        # Save new images in the output folder with modified filenames
        img_TL.save(os.path.join(output_folder, f"{base_filename}_TL.jpg"))
        img_TR.save(os.path.join(output_folder, f"{base_filename}_TR.jpg"))
        img_BL.save(os.path.join(output_folder, f"{base_filename}_BL.jpg"))
        img_BR.save(os.path.join(output_folder, f"{base_filename}_BR.jpg"))














##############################################################################################
# CALLING THE SCRIPT
##############################################################################################

if __name__ == '__main__':

    #############################################################################################
    # ARGPARSE
    ##############################################################################################

    parser = argparse.ArgumentParser()

    ####################################
    group = parser.add_argument_group('Dataset and paths')
    ####################################

    group.add_argument('--input_folder', type=str,
                    help='Directory or directories containing input regions')
    group.add_argument('--output_folder', type=str,
                    help='Directory or directories to save output regions')

    args = parser.parse_args()

    cut_images_in_folder(input_folder=args.input_folder, output_folder=args.output_folder)