""" This script is very similar to annotate_regions.py, but it is used to extract the cells from the images, and saving them as individual images. """

import argparse
import os
import sys
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

#############################################################################################
# IMPORTANT UTILITY FUNCTION
#############################################################################################

# type could be 'center-width-height' or 'top-left-bottom-right'
def extract_cells(image, label_df, save_dir, name, label_type='center-width-height', core_radius=3, conf_thres=None):
    """ Save the cells in an image with a label. The quantities in the label_df are normalized relative to the image width and height. """

    # if conf_thres is not None and label_type is 'center-width-height', raise a user warning
    if conf_thres is not None and label_type == 'center-width-height':
        print('User Warning: conf_thres is declared but the label_type is center-width-height. The conf_thres will be ignored.')

    # get the image width and height
    width, height = image.size

    # traverse over all rows in the label_df
    for row in list(label_df.iterrows()):

        # get the row as a dictionary
        row = row[1].to_dict()

        if label_type == 'center-width-height':
            # get the TL and BR coordinates
            TL_x = int((row['center_x'] - row['box_width']/2)*width)
            TL_y = int((row['center_y'] - row['box_height']/2)*height)
            BR_x = int((row['center_x'] + row['box_width']/2)*width)
            BR_y = int((row['center_y'] + row['box_height']/2)*height)

            # Make sure the coordinates are within the image
            TL_x = max(TL_x, 0)
            TL_y = max(TL_y, 0)
            BR_x = min(BR_x, width)
            BR_y = min(BR_y, height)

        elif label_type == 'top-left-bottom-right':
            # if the row has <6 keys, raise an error
            assert len(row.keys()) >= 6, f'row has less than 6 keys, please check the label file, do you mean center-width-height?'

            # only keep the rows with confidence above conf_thres
            if conf_thres is not None:
                if row['confidence'] < conf_thres:
                    continue

            # get the TL and BR coordinates
            TL_x = int((row['TL_x']))
            TL_y = int((row['TL_y']))
            BR_x = int((row['BR_x']))
            BR_y = int((row['BR_y']))

        # crop the image
        cell = image.crop((TL_x, TL_y, BR_x, BR_y))

        # save the image
        cell.save(os.path.join(save_dir, f'{name}_{TL_x}_{TL_y}_{BR_x}_{BR_y}_{int(row["class"])}_{round(row["confidence"],2)}.jpg'))

#############################################################################################
# MAIN
#############################################################################################

if __name__ == '__main__':

 
    #############################################################################################
    # ARGPARSE
    #############################################################################################

    parser = argparse.ArgumentParser()

    ####################################
    group = parser.add_argument_group('Dataset and paths')
    ####################################

    group.add_argument('--input_dir', type=str,
                        help='Directory or directories containing input images')
    group.add_argument('--labels_dir', type=str,
                        help='Directory or directories containing input labels')  
    group.add_argument('--output_dir', type=str,
                        help='Directory or directories to save output annotated images')  
    group.add_argument('--label_type', type=str, default='center-width-height',
                        help='Type of label. Could be center-width-height or top-left-bottom-right')
    group.add_argument('--image_ext', type=str, default='.jpg',
                        help='Extension of the images in the input_dir')
    group.add_argument('--conf_thres', type=float, default=None,
                        help='Confidence threshold for the output files. If None, all output files are used. Should only be used if the output files are in the format of YOLO outputs top-left-bottom-right.')

    args = parser.parse_args()

    # if conf_thres is not None and label_type is 'center-width-height', raise a user warning
    if args.conf_thres is not None and args.label_type == 'center-width-height':
        print('User Warning: conf_thres is declared but the label_type is center-width-height. The conf_thres will be ignored.')

    # if args.output_dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #  list all images in the input_dir, only .jpg files
    files = [file for file in os.listdir(args.input_dir) if file.endswith(args.image_ext)]

    # traverse over all files
    for file in tqdm(files):

        continuation = True
            
        # get the basename of the file
        basename = os.path.basename(file)

        # label is named as basename but with .txt extension
        label_path = os.path.join(args.labels_dir, os.path.splitext(basename)[0] + '.txt')

        # check if the label file is empty, if so, save the image as is and continue
        if os.stat(os.path.join(label_path)).st_size == 0:
            continuation = False

        if continuation:

            # open the image
            image = Image.open(os.path.join(args.input_dir, file))

            # open the label file as a pandas dataframe
            label_df = pd.read_csv(label_path, sep='\t', header=None)

            if args.label_type == 'center-width-height':

                # rename the columns of df to class, center_x, center_y, width, height
                label_df = label_df.rename(columns={0: 'class', 1: 'center_x', 2: 'center_y', 3: 'box_width', 4: 'box_height'})

                # extract the cells from the image
                extract_cells(image, label_df, args.output_dir, os.path.splitext(basename)[0], args.label_type, conf_thres=args.conf_thres)

            elif args.label_type == 'top-left-bottom-right':

                # rename the columns of df to TL_x, TL_y, BR_x, BR_y, confidence, class
                label_df = label_df.rename(columns={0: 'TL_x', 1: 'TL_y', 2: 'BR_x', 3: 'BR_y', 4: 'confidence', 5: 'class'})

                # extract the cells from the image
                extract_cells(image, label_df, args.output_dir, os.path.splitext(basename)[0], args.label_type, conf_thres=args.conf_thres)

            else:
                raise ValueError(f'label_type {args.label_type} is not supported. Please use center-width-height or top-left-bottom-right.')