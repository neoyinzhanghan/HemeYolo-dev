import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
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

    group.add_argument('--ai_labels_dir', type=str,
                        help='Directory or directories containing input labels')
    group.add_argument('--output_dir', type=str,
                        help='Directory or directories to save output labels')

    ####################################
    group = parser.add_argument_group('Hyperparameters')
    ####################################

    group.add_argument('--b_min', type=float, default=0,
                        help='Minimum value of the x-axis')
    group.add_argument('--b_max', type=float, default=0.5,
                        help='Maximum value of the x-axis')
    group.add_argument('--log_scale', type=bool, default=False,
                        help='Whether to use log scale for the y-axis')

    args = parser.parse_args()

    conf_lst = []
    # traverse through all (and only) .txt files in the labels_dir
    for label_file in tqdm([file for file in os.listdir(args.ai_labels_dir) if file.endswith('.txt')]):
        # if the file is empty, skip
        if os.stat(os.path.join(args.ai_labels_dir, label_file)).st_size == 0:
            continue

        else:
            # read the label file as a dataframe
            df = pd.read_csv(os.path.join(args.ai_labels_dir, label_file), sep='\t', header=None)

            # traverse through the dataframe rows
            for index, row in df.iterrows():

                # get the row as a dictionary
                row = row.to_dict()

                # get the second last entry of the row which is the confidence score
                conf = row[len(row) - 2]

                # append the confidence score to the conf_lst
                conf_lst.append(conf)

    # convert the conf_lst to a numpy array
    conf_lst = np.array(conf_lst)

    # plot the histogram of the confidence scores, the height of histogram is in log10 scale, focus on the range b_min to b_max, label the axes
    plt.hist(conf_lst, bins=100, log=args.log_scale, range=(args.b_min, args.b_max))
    plt.xlabel('Confidence Score')

    # label the y axes 'Number of Regions' and add (log scale) if log=True
    if args.log_scale:
        plt.ylabel('Number of Regions (log scale)')
    else:
        plt.ylabel('Number of Regions')

    # save the histogram as a png file to the output_dir, named confidence_histogram.png
    if args.log_scale:
        plt.savefig(os.path.join(args.output_dir, 'confidence_histogram_log.png'))
    else:
        plt.savefig(os.path.join(args.output_dir, 'confidence_histogram.png'))