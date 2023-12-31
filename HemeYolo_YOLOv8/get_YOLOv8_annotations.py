import os
from ultralytics import YOLO
import argparse
import glob
import pandas as pd
import torch
from tqdm import tqdm
import cv2

#############################################################################################
# THE SCRIPT
#############################################################################################

if __name__ == '__main__':

    #############################################################################################
    # ARGPARSE
    #############################################################################################

    parser = argparse.ArgumentParser()

    ####################################
    group = parser.add_argument_group('Dataset and paths')
    ####################################

    group.add_argument('--data_folder', type=str,
                        help='Directory or directories containing input image')
    group.add_argument('--output_folder', type=str,
                        help='Directory or directories to save output labels')
    # group.add_argument('--cfg_path', type=str,
    #                     help='Path to your config file')
    group.add_argument('--chkpt_path', type=str,
                        help='Path to your checkpoint file')

    ####################################
    group = parser.add_argument_group('Hyperparameters')
    ####################################

    group.add_argument('--conf_thres', type=float, default=0,
                       help='Confidence threshold for predictions')
    
    group.add_argument('--verbose', type=bool, default=False,
                       help='Whether or not you want the program to print a bunch of stuff that help you debug.')


    args = parser.parse_args()

    # if the output_folder does not exist, create it
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    def force_cudnn_initialization():
        s = 32
        dev = torch.device('cuda')
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

    force_cudnn_initialization

    # Load the YOLO model
    print('Loading model...')
    model = YOLO(args.chkpt_path)

    # Get a list of all the images in the data_folder with file extensions .jpg and .png
    # The glob.glob() function returns a list of paths matching a pathname pattern
    # The os.path.join() function joins one or more path components

    print('Grabbing image paths...')
    images = glob.glob(os.path.join(args.data_folder, '*.jpg')) + glob.glob(os.path.join(args.data_folder, '*.png'))

    for image_path in tqdm(images):
        
        # # grab the image from the image_path as a np array
        # image = cv2.imread(image_path)

        # print(image_path)

        result = model([image_path], conf=args.conf_thres)[0]

        boxes = result.boxes.data  # Boxes object for bbox outputs

        if args.verbose:
            print(boxes)

        ### Start a pandas dataframe to store the result, saving each box and corresponding probability as a row
        ### The row is just concatenated to the dataframe. Each row is the just the box and the probability, and a dummie class index 0
        ### Assuming that box is in the format ['TL_x', 'TL_y', 'BR_x', 'BR_y', 'confidence', 'class']

        df = pd.DataFrame(columns=['TL_x', 'TL_y', 'BR_x', 'BR_y', 'confidence', 'class'])

        l1 = len(boxes)

        for i in range(l1):
            box = boxes[i]

            TL_x, TL_y = int(box[0]), int(box[1])
            BR_x, BR_y = int(box[2]), int(box[3])
            conf = float(box[4])
            cls = int(box[5])

            # use pd.concat instead of append to avoid deprecation
            df = pd.concat([df, pd.DataFrame([[TL_x, TL_y, BR_x, BR_y, conf, cls]], columns=['TL_x', 'TL_y', 'BR_x', 'BR_y', 'confidence', 'class'])])

        # Save the dataframe as a .txt, no header and no index, and no column name, nothing, separated by \t
        # the name of the file is the same as the image file but with .txt extension
        df.to_csv(os.path.join(args.output_folder, os.path.basename(image_path)[:-4] + '.txt'), sep='\t', header=None, index=None)