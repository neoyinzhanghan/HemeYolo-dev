import pandas as pd
import os
import matplotlib.pyplot as plt

def _calculate_iou(box1, box2):
    """ Assume that box1 is an array of [TL_x, TL_y, BR_x, BR_y] and box2 is an array of [TL_x, TL_y, BR_x, BR_y],
    Both have be in relative coordinates. """
    
    # Check the relationship between TL and BR coordinates, if not correct, raise a ValueError
    if box1[0] > box1[2] or box1[1] > box1[3]:
        raise ValueError(f"Box1 coordinates {box1} are not in the correct format")
    if box2[0] > box2[2] or box2[1] > box2[3]:
        raise ValueError(f"Box2 coordinates {box2} are not in the correct format")
    
    # Calculate the area of the intersection
    intersection_area = (min(box1[2], box2[2]) - max(box1[0], box2[0])) * (min(box1[3], box2[3]) - max(box1[1], box2[1]))
    print(intersection_area)

    # Calculate the area of the union
    union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection_area
    print(union_area)

    print(box1)
    print(box2)

    # Calculate the iou
    iou = intersection_area / union_area

    return iou

def level_p_coverage(label_path, output_path, conf_level=0, iou_level=0.75, region_width=512, region_height=512):
    """ For every box in the label_path, check if it is covered by a box in the output_path. For a given box ...
    Traverse through all boxes in the file of the output_path with confidence above conf_level, convert to relative coordinates using region_width and region_height,
    if the box in the output path has an iou with the box in label_path above iou_level, then increment the coverage counter by 1.
    At the end of the function, return the coverage counter divided by the number of boxes in the label_path.
    """

    # Read in the label_path file as a pandas data frame, rename the columns as [class, center_x, center_y, box_width, box_height]
    # if label_path is empty, return 1
    if os.stat(label_path).st_size == 0:
        return 1
    
    label_df = pd.read_csv(label_path, sep="\t", header=None)
    label_df.columns = ["class", "center_x", "center_y", "box_width", "box_height"]

    # Create a new data frame with the processed label data with columns named [TL_x, TL_y, BR_x, BR_y, confidence, class]
    # where TL_x = (center_x - box_width/2), the rest you get the idea
    label_df_processed = pd.DataFrame()
    label_df_processed["TL_x"] = (label_df["center_x"] - label_df["box_width"] / 2)
    label_df_processed["TL_y"] = (label_df["center_y"] - label_df["box_height"] / 2) 
    label_df_processed["BR_x"] = (label_df["center_x"] + label_df["box_width"] / 2)
    label_df_processed["BR_y"] = (label_df["center_y"] + label_df["box_height"] / 2)
    label_df_processed["confidence"] = 1
    label_df_processed["class"] = label_df["class"]

    # Read in the output_path file as a pandas data frame, rename the columns as [TL_x, TL_y, BR_x, BR_y, confidence, class]
    output_df = pd.read_csv(output_path, sep="\t", header=None)
    output_df.columns = ["TL_x", "TL_y", "BR_x", "BR_y", "confidence", "class"]

    # Traverse through all boxes in the label_path file
    coverage_counter = 0
    for index, row in label_df_processed.iterrows():
        # get the row as a dictionary
        row = row.to_dict()

        # read the box in the label_path file
        label_box = [row["TL_x"]*region_width, row["TL_y"]*region_height, row["BR_x"]*region_width, row["BR_y"]*region_height]

        # Traverse through all boxes in the output_path file with confidence above conf_level
        for index, row in output_df[output_df["confidence"] > conf_level].iterrows():
            # get the row as a dictionary
            row = row.to_dict()

            # read the box in the output_path file
            output_box = [row["TL_x"], row["TL_y"], row["BR_x"], row["BR_y"]]

            # calculate the iou between the two boxes
            iou = _calculate_iou(label_box, output_box)

            # if the iou is above iou_level, increment the coverage counter by 1
            if iou > iou_level:
                coverage_counter += 1
                break
    
    # return the coverage counter divided by the number of boxes in the label_path
    return coverage_counter / len(label_df_processed)

if __name__ == "__main__":

    labels_dir = "/Users/neo/Documents/Research/DeepHeme/HemeYolo_data/data3/split/valid/labels"
    outputs_dir = "/Users/neo/Documents/Research/DeepHeme/HemeYolo_data/data3/split/valid/YOLO_outputs"

    # Get a list of label_path in the labels_dir with extension .txt
    # Get a list of output_path in the outputs_dir with extension .txt
    label_paths = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith(".txt")]
    output_paths = [os.path.join(outputs_dir, file) for file in os.listdir(outputs_dir) if file.endswith(".txt")]

    if len(label_paths) != len(output_paths):
        raise ValueError("The number of label files and output files are not equal")
    
    coverage_list = []
    # Traverse through all label_paths and output_paths

    boxes_count = []

    for i in range(len(label_paths)):
        # Get the label_path and output_path
        label_path = label_paths[i]
        output_path = output_paths[i]

        # Calculate the level_p_coverage
        coverage = level_p_coverage(label_path, output_path)

        coverage_list.append(coverage)

        # if the label_path is empty, boxes_count append 0
        if os.stat(label_path).st_size == 0:
            boxes_count.append(0)
            continue
        else:
            label_df = pd.read_csv(label_path, sep="\t", header=None) 
            label_df.columns = ["class", "center_x", "center_y", "box_width", "box_height"]

            boxes_count.append(len(label_df))
    
    # Plot a histogram of the coverage_list
    plt.hist(coverage_list, bins=20)

    # Calculate the region level average coverage
    region_level_average_coverage = sum(coverage_list) / len(coverage_list)

    # Calculate the cell_level coverage
    for label_path in label_paths:
        # Read in the label_path file as a pandas data frame, rename the columns as [class, center_x, center_y, box_width, box_height]
        label_df = pd.read_csv(label_path, sep="\t", header=None)
        label_df.columns = ["class", "center_x", "center_y", "box_width", "box_height"]

        boxes_count.append(len(label_df))

    # Use the boxes_count to get a weighted average of the coverage_list
    weighted_average_coverage = sum([coverage_list[i] * boxes_count[i] for i in range(len(coverage_list))]) / sum(boxes_count)

    # Add the region_level_average_coverage and weighted_average_coverage to the plot
    plt.axvline(region_level_average_coverage, color="red", linestyle="dashed", linewidth=1)
    plt.axvline(weighted_average_coverage, color="green", linestyle="dashed", linewidth=1)

    # Add the text to the plot
    plt.text(region_level_average_coverage, 0, "Region level average coverage", rotation=90)
    plt.text(weighted_average_coverage, 0, "Weighted average coverage", rotation=90)

    # Label the plot
    plt.xlabel("Coverage Proportion")
    plt.ylabel("Frequency")

    # display the plot
    plt.show()

    
