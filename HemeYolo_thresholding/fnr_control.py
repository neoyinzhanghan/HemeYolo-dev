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

def Hoeffding_p_value(alpha:float, fnr:float, total:int) -> float:
    """ Return the Hoeffding p-value given alpha and the false negative rate, and the total number of labels.
    p_c = \exp \ (-2n (\alpha-\hat{R}_c)_+^2), where \hat{R}_c is the empirical false negative rate. """

    # calculate the Hoeffding p-value, make sure to ReLU the difference between alpha and fnr before squaring
    p_value = np.exp(-2 * total * (max(alpha - fnr, 0))**2)

    # return the Hoeffding p-value
    return p_value


def find_threshold(label_dir:str, output_dir:str, alpha:float, max_p_value:float=0.05, min_iou:float=0.5, region_width:int=512, region_height:int=512, grid_size=100):
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
        p_value = Hoeffding_p_value(alpha, fnr, total)

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

def plot_p_values(label_dir:str, output_dir:str, alpha:float, max_p_value:float=0.05, min_iou:float=0.5, region_width:int=512, region_height:int=512, grid_size=100):
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
        p_value = Hoeffding_p_value(alpha, fnr, total)

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
    calibrate = True   
    test = False

    label_dir = "/Users/neo/Documents/Research/DeepHeme/HemeYolo_data/data4/split/train/labels"
    output_dir = "/Users/neo/Documents/Research/DeepHeme/HemeYolo_data/data4/split/train/YOLO_outputs"

    if calibrate: # calibration is done on the validation set trying to find the threshold that achieves the desired false negative rate with p-value at a desired significance level

        alpha = 0.05
        max_p_value = 0.05
        min_iou = 0.5

        threshold, fnr, p_value = find_threshold(label_dir, output_dir, alpha, max_p_value, min_iou)
        print(f'The threshold that achieves the desired false negative rate {fnr} (less than {alpha}) with p-value {p_value} less than {max_p_value} is {threshold}')

        plot_p_values(label_dir, output_dir, alpha, max_p_value, min_iou)

    if test: # test mode is done on the test set to calculate the false negative rate and the p-value given a threshold

        alpha = 0.05
        threshold = 0.5
        min_iou = 0.5
        threshold = 0.12121212121212122

        fnr, total = FNR(label_dir, output_dir, threshold, min_iou)

        p_value = Hoeffding_p_value(alpha, fnr, total)

        print(f'The false negative rate is {fnr} and the p-value is {p_value} given threshold {threshold} with total number of labels {total}')

        
