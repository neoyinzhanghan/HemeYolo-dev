from update_csv import get_corners
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import argparse


##############################################################################################
# THE SCRIPT
##############################################################################################


if __name__ == '__main__':

    ##############################################################################################
    # ARGPARSE
    ##############################################################################################

    parser = argparse.ArgumentParser()

    ####################################
    group = parser.add_argument_group('Dataset and paths')
    ####################################

    group.add_argument('--images_dir', type=str,
                        help='Directory or directories containing input images')
    group.add_argument('--labels_dir', type=str,
                        help='Directory or directories containing input labels')
    group.add_argument('--label_type', type=str, default='top-left-bottom-right',
                        help='Type of label. Could be center-width-height or top-left-bottom-right')
    group.add_argument('--output_dir', type=str,
                        help='Directory or directories to save output cropped images')
    group.add_argument('--image_ext', type=str, default='.png',
                        help='File extension of images')

    args = parser.parse_args()

    # if args.output_dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Grab all image files in images_dir, with extension .jpg or .png
    image_files = [os.path.join(args.images_dir, image_name) for image_name in os.listdir(args.images_dir) if image_name.endswith('.jpg') or image_name.endswith('.png')]

    # Traverse through the images_dir, use the corresponding label in the labels_dir to crop out the regions indicated by the label and save them in the output_dir
    for image_name in tqdm(image_files):

        # get the image path
        image_path = os.path.join(args.images_dir, image_name)

        # open the image
        image = Image.open(image_path)

        # get the image width and height
        width, height = image.size  

        image_base_name = os.path.basename(image_name)
        # get the label path
        label_path = os.path.join(args.labels_dir, image_base_name.replace(args.image_ext, '.txt'))

        # if the label file is empty, skip
        if os.stat(label_path).st_size == 0:
            continue

        else:
            # read the label file as a dataframe
            df = pd.read_csv(label_path, sep='\t', header=None)

            if args.label_type == 'center-width-height':
                # convert the df to contain only the columns we need including 'center_x', 'center_y', 'box_width', 'box_height'

                df = df[[0, 1, 2, 3, 4]]
                df.columns = ['class', 'center_x', 'center_y', 'box_width', 'box_height']

                # traverse through the dataframe rows
                for index, row in df.iterrows():             
                    # get the row as a dictionary
                    row = row.to_dict()

                    # Only access square boxes
                    assert row['box_width'] == row['box_height'], f'box_width and box_height are not equal for a box requested in {image_name}'

                    # Region must be a square
                    assert width == height, f'region width and region height are not equal for {image_name}'

                    radius = row['box_width'] / 2

                    # get the TL and BR coordinates
                    TL_corner, BR_corner = get_corners(row['center_x']*width, row['center_y']*height, radius*width, width, height)

                    # crop the image
                    cropped_image = image.crop((TL_corner[0], TL_corner[1], BR_corner[0], BR_corner[1]))

                    # save the cropped image
                    cropped_image.save(os.path.join(args.output_dir, image_base_name.replace(args.image_ext, f'_{index}{args.image_ext}')))
            
            elif args.label_type == 'top-left-bottom-right':

                # rename the df columns as [[x1, y1, x2, y2, confidence, class]]
                df.columns = ['TL_x', 'TL_y', 'BR_x', 'BR_y', 'confidence', 'class']

                # traverse through the dataframe rows
                for index, row in df.iterrows():
                    # get the row as a dictionary
                    row = row.to_dict()

                    # crop the image
                    cropped_image = image.crop((row['TL_x'], row['TL_y'], row['BR_x'], row['BR_y']))

                    # save the cropped image
                    cropped_image.save(os.path.join(args.output_dir, image_base_name.replace(args.image_ext, f'_{index}{args.image_ext}')))
