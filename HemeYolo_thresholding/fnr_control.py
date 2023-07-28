""" The purpose of the script is to control the false negative rate of the HemeYolo model. 
-- the script takes the output of the HemeYolo model and the ground truth and calculates the false negative rate
-- the script then calculates the Hoeffding bound to the probability of observing the false negative rate assuming the true false negative rate is alpha (p-value)
-- the script then pick the highest threshold that achieves the desired false negative rate with p-value less than 0.05

-- the variable referencing path to the input folder for ground truth will be called "label_dir"
-- the variable referencing path to the input folder for HemeYolo output will be called "output_dir"
-- the variable referencing path to the input image folder will be called "image_dir"
-- the labels are saved in \\t separated text files with the following columns: class, center_x, center_y, width, height, in relative coordinates
-- the output is saved in \\t separated text files with the following columns: TL_x, TL_y, BR_x, BR_y, confidence, class, in absolute coordinates
-- the input images are technically needed to convert the output from absolute coordinates to relative coordinates, but we assume that users will know the image size

-- the variable for the desired false negative rate will be called "alpha"
-- the variable for the desired p-value will be called "p_value"
-- the variable for the output threshold will be called "threshold"
-- the variable for the IoU threshold will be called "min_iou"
"""

import os
import numpy as np
from HemeYolo_data.utils import get_label_as_df, get_output_as_df
from HemeYolo_thresholding.calculate_iou import bb_intersection_over_union as iou
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse


def FNR(label_dir:str, output_dir:str, threshold:float, min_iou:float=0.5, region_width:int=512, region_height:int=512) -> float:
    """ Calculate the false negative rate of the HemeYolo model. Return the false negative rate, and the number of labels."""

    # get a list of label paths from the labels_dir make sure file extension is txt
    label_names = [os.path.join(label_dir, label_name) for label_name in os.listdir(label_dir) if os.path.splitext(label_name)[1] == '.txt']
    label_names = [os.path.basename(label_name) for label_name in label_names]

    # get a list of output paths from the output_dir make sure file extension is txt
    output_names = [os.path.join(output_dir, output_name) for output_name in os.listdir(output_dir) if os.path.splitext(output_name)[1] == '.txt']
    output_names = [os.path.basename(output_name) for output_name in output_names]

    # check that each label has a corresponding output
    for label_name in label_names:
        if label_name not in output_names:
            raise ValueError('Missing output for label: {0}'.format(label_name))
    
    # initialize the false negative rate
    fn_count = 0
    total = 0

    # for each label, count the number of false negatives
    for label_name in label_names:

        found = False

        # get the label and output as pandas dataframes
        label_df = get_label_as_df(os.path.join(label_dir, label_name))
        # if the label is empty, then skip
        if label_df is None:
            continue

        output_df = get_output_as_df(os.path.join(output_dir, label_name))
        # if the output is empty, then all the labels are false negatives, but we are assuming that the output is not empty for now, raise an error if it is
        if output_df is None:
            raise ValueError('Missing output for label: {0}. This should not be happening as we are assuming the output txt files are generated with confidencen threshold 0'.format(label_name))

        # Trim the output_df by removing all rows with confidence below threshold
        output_df = output_df[output_df['confidence'] > threshold]

        # calculate the TL and BR coordinates of the label in relative coordinates
        label_df['TL_x'] = label_df['center_x'] - label_df['box_width'] / 2
        label_df['TL_y'] = label_df['center_y'] - label_df['box_height'] / 2

        label_df['BR_x'] = label_df['center_x'] + label_df['box_width'] / 2
        label_df['BR_y'] = label_df['center_y'] + label_df['box_height'] / 2

        # calculate the TL and BR coordinates of the label_df in absolute coordinates
        label_df['TL_x'] = label_df['TL_x'] * region_width
        label_df['TL_y'] = label_df['TL_y'] * region_height

        label_df['BR_x'] = label_df['BR_x'] * region_width
        label_df['BR_y'] = label_df['BR_y'] * region_height

        # for each label, check if there is an output that has IoU > min_iou
        for index, row in label_df.iterrows():
            # get row as a dictionary
            row_dict = row.to_dict()

            # get the label coordinates
            label_TL_x = row_dict['TL_x']
            label_TL_y = row_dict['TL_y']
            label_BR_x = row_dict['BR_x']
            label_BR_y = row_dict['BR_y']
            # boxA will be the label box
            boxA = [label_TL_x, label_TL_y, label_BR_x, label_BR_y]

            # iterate through all the output boxes
            for index, row in output_df.iterrows():
                # get row as a dictionary
                row_dict = row.to_dict()

                # get the output coordinate
                output_TL_x = row_dict['TL_x']
                output_TL_y = row_dict['TL_y']
                output_BR_x = row_dict['BR_x']
                output_BR_y = row_dict['BR_y']
                # boxB will be the output box
                boxB = [output_TL_x, output_TL_y, output_BR_x, output_BR_y]

                # calculate the IoU between the label box and all the output boxes
                IoU = iou(boxA, boxB)

                # if there is an output box with IoU > min_iou, then this label is not a false negative, add 1 to the total count but do not increment the false negative count and break
                if IoU > min_iou:
                    total += 1
                    found = True
                    break

            if found:
                continue
            
            else:
                # if there is no output box with IoU > min_iou, then this label is a false negative, add 1 to the total count and the false negative count
                fn_count += 1
                total += 1
    
    # return the false negative rate
    return fn_count / total , total

def Hoeffding_p_value(alpha:float, fnr:float, total:int, raw:bool=False) -> float:
    """ Return the Hoeffding p-value given alpha and the false negative rate, and the total number of labels.
    p_c = \exp \ (-2n (\alpha-\hat{R}_c)_+^2), where \hat{R}_c is the empirical false negative rate. """

    if not raw:
        # calculate the Hoeffding p-value, make sure to ReLU the difference between alpha and fnr before squaring
        p_value = np.exp(-2 * total * (max(alpha - fnr, 0))**2)
    else:
        p_value = fnr

    # return the Hoeffding p-value
    return p_value


def find_threshold(label_dir:str, output_dir:str, alpha:float, max_p_value:float=0.05, min_iou:float=0.5, region_width:int=512, region_height:int=512, grid_size=100, raw=False):
    """ Find the threshold that achieves the desired false negative rate with p-value less than 0.05. Return the threshold, the false negative rate, and the p-value.
    Return the threshold, the false negative rate, and the p-value."""

    # initialize the threshold
    threshold = 0

    # initialize the false negative rate
    observed_fnr = 1

    # initialize the p-value
    observed_p_value = 1

    # initialize the total number of labels
    total = 0

    # initialize the grid, back traverse from 1 to 0
    grid = np.linspace(0, 1, grid_size)

    # for each threshold in the grid, calculate the false negative rate and the p-value, traverse from 0 to 1
    # make sure to traverse from 0 to 1, because the false negative rate is monotonically increasing with threshold
    # and the p-value is monotonically increasing with threshold
    # the largest threshold that achieves the desired false negative rate with p-value less than 0.05 will be the output threshold
    for t in tqdm(grid):
        # calculate the false negative rate
        fnr, total = FNR(label_dir, output_dir, t, min_iou, region_width, region_height)

        # calculate the p-value
        p_value = Hoeffding_p_value(alpha, fnr, total, raw=raw)

        # if the p-value is less than 0.05, then update the threshold and the p-value and continue
        if p_value < max_p_value:
            threshold = t
            observed_p_value = p_value
            observed_fnr = fnr

            continue

        if p_value >= max_p_value:
            break

    # return the threshold, the false negative rate, and the p-value
    return threshold, observed_fnr, observed_p_value

def plot_p_values(label_dir:str, output_dir:str, alpha:float, max_p_value:float=0.05, min_iou:float=0.5, region_width:int=512, region_height:int=512, grid_size=100, raw=False):
    """ Plot the p-values for different thresholds. """

    # initialize the grid, back traverse from 0 to 1
    grid = np.linspace(0, 1, grid_size)

    # initialize the p-values
    p_values = []

    # for each threshold in the grid, calculate the false negative rate and the p-value, traverse from 0 to 1
    # make sure to traverse from 0 to 1, because the false negative rate is monotonically increasing with threshold
    # and the p-value is monotonically increasing with threshold
    # the largest threshold that achieves the desired false negative rate with p-value less than 0.05 will be the output threshold
    for t in tqdm(grid):
        # calculate the false negative rate
        fnr, total = FNR(label_dir, output_dir, t, min_iou, region_width, region_height)

        # calculate the p-value
        p_value = Hoeffding_p_value(alpha, fnr, total, raw=raw)

        # append the p-value to the list
        p_values.append(p_value)

    # Use Seaborn for better aesthetics
    sns.set(style='whitegrid')

    # create figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # plot the p-values
    ax.plot(grid, p_values, linewidth=2, color='blue', alpha=0.7)
    
    # add a line at y = 0.05
    ax.axhline(y=max_p_value, color='r', linestyle='--', linewidth=2, alpha=0.7)

    # Add labels and title
    ax.set_xlabel('Threshold', fontsize=14)
    ax.set_ylabel('p-value', fontsize=14)
    ax.set_title('p-value vs. Threshold', fontsize=16, fontweight='bold')

    # Set the x and y axis limits
    ax.set_xlim([min(grid), max(grid)])
    ax.set_ylim([0, max(p_values)])

    # Add a grid
    ax.grid(True)

    # Add a legend
    ax.legend(['p-values', f'Significance level ({max_p_value})'], loc='upper right')

    # Remove the top and right spines from plot
    sns.despine()

    # Display the plot
    plt.show()


if __name__ == '__main__':

    #############################################################################################################################
    # ARGPARSE
    #############################################################################################################################

    parser = argparse.ArgumentParser()

    ####################################
    group = parser.add_argument_group('Dataset and paths')
    ####################################

    group.add_argument('--label_dir', type=str,
                        help='Directory containing input labels')
    group.add_argument('--annotation_dir', type=str,
                        help='Directory containing HemeYolo annotations')
    
    ####################################
    group = parser.add_argument_group('Hyperarameters')
    ####################################
    group.add_argument('--mode', type=int, default=1,
                        help='Mode of operation. 1 for calibration, 0 for test')
    group.add_argument('--alpha', type=float, default=None,
                        help='Desired false negative rate')
    group.add_argument('--max_p_value', type=float, default=None,
                        help='Desired p-value')
    group.add_argument('--min_iou', type=float, default=None,
                        help='IoU threshold')
    group.add_argument('--conf_thres', type=float, default=None,
                        help='Threshold for which you want to test the FNR performance')
    group.add_argument('--region_width', type=int, default=512,
                        help='Width of the region')
    group.add_argument('--region_height', type=int, default=512,
                        help='Height of the region')


    args = parser.parse_args()

    #############################################################################################################################
    # THE SCRIPT
    #############################################################################################################################

    # min_iou must be declared in all cases, as a float between 0 and 1, if violated, raise a ValueError
    if args.min_iou is None:
        raise ValueError(f'min_iou must be declared.')
    if args.min_iou < 0 or args.min_iou > 1:
        raise ValueError(f'min_iou {args.min_iou} must be between 0 and 1.')
    
     # alpha must be declared in all cases, as a float between 0 and 1, if violated, raise a ValueError
    if args.alpha is None:
        raise ValueError(f'alpha must be declared.')
    if args.alpha < 0 or args.alpha > 1:
        raise ValueError(f'alpha {args.alpha} must be between 0 and 1.')

    # If the mode is 0 then max_p_value should be default values
    # If violated, print a UserWarning to inform the user that these parameters will be ignored
    if args.mode == 0:
        if args.max_p_value is not None:
            print('User Warning: max_p_value is declared but the mode is 1. The max_p_value will be ignored.')
        if args.min_iou is not None:
            print('User Warning: min_iou is declared but the mode is 1. The min_iou will be ignored.')

    # If the mode is 0, then the threshold must be declared, as a float between 0 and 1
    # If violated, raise a ValueError
    if args.mode == 0:
        if args.conf_thres is None:
            raise ValueError(f'Threshold must be declared when mode is 0.')
        if args.conf_thres < 0 or args.conf_thres > 1:
            raise ValueError(f'Threshold {args.conf_thres} must be between 0 and 1.')

    if args.mode == 1: # calibration is done on the validation set trying to find the threshold that achieves the desired false negative rate with p-value at a desired significance level

        threshold, fnr, p_value = find_threshold(args.label_dir, args.annotation_dir, args.alpha, args.max_p_value, args.min_iou, region_width=args.region_width, region_height=args.region_height, raw=True)
        print(f'The threshold that achieves the desired false negative rate {fnr} (less than {args.alpha}) with p-value {p_value} less than {args.max_p_value} is {threshold} when min_iou is {args.min_iou}')

        plot_p_values(args.label_dir, args.annotation_dir, args.alpha, args.max_p_value, args.min_iou, region_height=args.region_height, region_width=args.region_width, raw=True)

    if args.mode == 0: # test mode is done on the test set to calculate the false negative rate and the p-value given a threshold

        fnr, total = FNR(args.label_dir, args.annotation_dir, args.conf_thres, args.min_iou, region_width=args.region_width, region_height=args.region_height)

        p_value = Hoeffding_p_value(args.alpha, fnr, total)

        print(f'The false negative rate is {fnr} and the p-value is {p_value} given threshold {args.conf_thres} with total number of labels {total} when min_iou is {args.min_iou}')