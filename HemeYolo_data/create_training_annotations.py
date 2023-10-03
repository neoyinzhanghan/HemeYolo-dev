import os
import shutil
import pandas as pd
from tqdm import tqdm

annotations_dir = "/Users/neo/Documents/Research/DeepHeme/HemeYolo_data/PB_YOLO_2k/annotations_raw"
save_dir = "/Users/neo/Documents/Research/DeepHeme/HemeYolo_data/PB_YOLO_2k/PB_YOLO_2k/labels"

# if the save_dir does not exist, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# the annotations dir contains a bunch of txt tiles, comma separated, each line is a centroid
# the images are 2048 by 2048 and the annotations are the pixel coordinates of the centroids

# create a new annotation df with columns ['class', 'center_x', 'center_y', 'box_width', 'box_height'] in relative coordinates
# save the new annotation df as a txt file in the save_dir with the same name as the original file

# get the list of files in the annotations_dir
files = [file for file in os.listdir(annotations_dir) if file.endswith(".txt")]

for file in tqdm(files):
    new_df = []

    # if the file is empty, copy the file to the save_dir using shutil.copy under same file name and continue
    if os.stat(os.path.join(annotations_dir, file)).st_size == 0:
        shutil.copy(os.path.join(annotations_dir, file), os.path.join(save_dir, file))
        continue

    # traverse through each line of the file
    with open(os.path.join(annotations_dir, file), "r") as f:
        for line in f:
            # split the line by comma
            line = line.split(",")

            # convert the line to a dictionary
            line_dict = {
                "class": 0,
                "center_x": int(line[0]) / 2048,
                "center_y": int(line[1]) / 2048,
                "box_width": 96 / 2048,
                "box_height": 96 / 2048,
            }

            # append the line_dict to the new_df
            new_df.append(line_dict)

    # convert the new_df to a dataframe
    new_df = pd.DataFrame(new_df)

    # save the new_df as a txt file in the save_dir
    new_df.to_csv(os.path.join(save_dir, file), header=False, index=False, sep="\t")
