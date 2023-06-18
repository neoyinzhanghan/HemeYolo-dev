import os
from ultralytics import YOLO
import argparse
import glob
import pandas as pd
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
    # group.add_argument('--conf_thres', type=float, default=0,
    #                     help='Confidence threshold for predictions')
    # group.add_argument('--verbosity', type=int, default=0,
    #                     help='Verbosity level')

    args = parser.parse_args()

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

        result = model([image_path])[0]

        boxes = result.boxes  # Boxes object for bbox outputs
        probs = result.probs  # Class probabilities for classification outputs

        ### Start a pandas dataframe to store the result, saving each box and corresponding probability as a row
        ### The row is just concatenated to the dataframe. Each row is the just the box and the probability, and a dummie class index 0
        ### Assuming that box is in the format [TL_x, TL_y, BR_x, BR_y]

        df = pd.DataFrame(columns=['TL_x', 'TL_y', 'BR_x', 'BR_y', 'probability', 'class'])

        # if probs is None:
        #     # write an empty txt file
        #     df.to_csv(os.path.join(args.output_folder, os.path.basename(image_path).split('.')[0] + '.txt'), header=False, index=False, sep='\t')
        #     continue

        l1 = len(boxes)
        print(boxes)
        # l2 = len(probs)
        # print(probs)

        # if l2 != l1:
        #     raise ValueError('The number of boxes (got {l1}) and the number of probabilities (got {l2}) are not equal!')
        
        for i in range(l1):
            box = boxes[i]

            TL_x, TL_y = box[0], box[1]
            BR_x, BR_y = box[2], box[3]

            # use pd.concat instead of append to avoid deprecation
            df = pd.concat([df, pd.DataFrame([[TL_x, TL_y, BR_x, BR_y, 0]], columns=['TL_x', 'TL_y', 'BR_x', 'BR_y', 'class'])])

        # Save the dataframe as a .txt, no header and no index, and no column name, nothing, separated by \t
        df.to_csv(os.path.join(args.output_folder, os.path.basename(image_path).split('.')[0] + '.txt'), header=False, index=False, sep='\t')