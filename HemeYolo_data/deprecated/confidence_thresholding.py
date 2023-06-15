import os
import pandas as pd
from tqdm import tqdm
import argparse

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

    group.add_argument('--labels_dir', type=str,
                        help='Directory or directories containing input labels')
    group.add_argument('--output_dir', type=str,
                        help='Directory or directories to save output labels')

    ####################################
    group = parser.add_argument_group('Hyperparameters')
    ####################################

    group.add_argument('--conf_thres', type=float,
                        help='Confidence threshold')

    args = parser.parse_args()

    # create a new folder named 'confidence_thresholded' + conf_thres in the output_dir if it doesn't exist
    output_dir = os.path.join(args.output_dir, f'confidence_thresholded_{args.conf_thres}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # traverse through the labels_dir for only .txt files
    for label_file in tqdm([file for file in os.listdir(args.labels_dir) if file.endswith('.txt')]):
        
        # if the file is empty, save the file into the output_dir, file name appending _conf_thres.txt
        if os.stat(os.path.join(args.labels_dir, label_file)).st_size == 0:

            # save the file into the output_dir, file name appending _conf_thres.txt
            with open(os.path.join(output_dir, label_file.replace('.txt', f'_{args.conf_thres}.txt')), 'w') as f:
                pass

        else:   
            # read the label file as a dataframe
            df = pd.read_csv(os.path.join(args.labels_dir, label_file), sep='\t', header=None)

            # rename the columns to [[x1, y1, x2, y2, confidence, class]]
            df.columns = ['x1', 'y1', 'x2', 'y2', 'confidence', 'class']

            # filter out the rows with confidence < conf_thres
            df = df[df['confidence'] >= args.conf_thres]

            # save the dataframe into the output_dir using the same name as the original file do not save column names or row names
            df.to_csv(os.path.join(output_dir, label_file), sep='\t', header=False, index=False)