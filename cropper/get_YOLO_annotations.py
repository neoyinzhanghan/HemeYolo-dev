import os
import cv2
from pytorchyolo import detect, models
import argparse
import glob
from tqdm import tqdm
import pandas as pd


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
    group.add_argument('--cfg_path', type=str,
                        help='Path to your config file')
    group.add_argument('--chkpt_path', type=str,
                        help='Path to your checkpoint file')
    group.add_argument('--conf_thres', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    group.add_argument('--verbosity', type=int, default=0,
                        help='Verbosity level')

    args = parser.parse_args()

    # Load the YOLO model
    model = models.load_model(args.cfg_path, args.chkpt_path)

    # traverse through all images in the data_folder with extension .jpg and .png using a tqdm progress bar

    for image_path in tqdm(glob.glob(os.path.join(args.data_folder, '*.jpg')) + glob.glob(os.path.join(args.data_folder, '*.png'))):

        # Load the image as a numpy array
        img = cv2.imread(image_path)

        # Convert OpenCV bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Runs the YOLO model on the image 
        boxes = detect.detect_image(model, img, conf_thres=args.conf_thres)

        # Output will be a numpy array in the following format:
        # [[x1, y1, x2, y2, confidence, class]]

        # Save the output in the output_folder as a .txt file with the same name as the image file with no columns names
        # For instance, if your image is named image.jpg, the output will be saved as image.txt
        # Each row in the .txt file will be in the following format (tab separated):
        # class x_1 y_1 x_2 y_2 confidence

        # Create the output_folder if it doesn't exist
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        
        # Save the output
        output_path = os.path.join(args.output_folder, os.path.basename(image_path).split('.')[0] + '.txt')

        # convert the boxes which is a numpy array to a pandas dataframe and then to csv
        boxes = pd.DataFrame(boxes)

        if args.verbosity == 1:
            print(boxes) # for debugging

        boxes.to_csv(output_path, sep='\t', index=False, header=False)