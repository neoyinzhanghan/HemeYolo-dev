##############################################################################################
# PACKAGE IMPORT
##############################################################################################

import os
import pandas as pd
import argparse
from tqdm import tqdm
import random

# set random seed
random.seed(42)


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
                    help='Directory or directories containing input images')
    group.add_argument('--labels_dir', type=str,
                    help='Directory or directories to save output labels')
    group.add_argument('--save_dir', type=str,
                    help='Directory to save the custom dataset')

    ####################################
    group = parser.add_argument_group('Hyperparameters')
    ####################################

    group.add_argument('--training_prop', type=float, default=0.8,
                    help='The proportion of data to be used for training')

    args = parser.parse_args()

    # Create folder named "custom" if it doesn't exist
    custom_path = os.path.join(args.save_dir, 'custom')
    
    if not os.path.exists(custom_path):
        os.makedirs(custom_path)

    classes_names_path = os.path.join(custom_path, 'classes.names')
    with open(classes_names_path, 'w') as f:
        f.write('WBC')

    # Split data in the data_folder into training and validation sets and create a .txt file for each set in the format data/custom/images/image_name.jpg
    # For instance, if you have 100 images in your data_folder, 80 will be used for training and 20 for validation
    # The .txt files will be named train.txt and valid.txt
    
    # Get all image names
    image_names = [image_name for image_name in os.listdir(args.data_folder) if image_name.endswith('.jpg')]

    # Shuffle image names
    random.shuffle(image_names)

    # Split image names into training and validation sets
    training_image_names = image_names[:int(len(image_names) * args.training_prop)]
    validation_image_names = image_names[int(len(image_names) * args.training_prop):]

    # Create train.txt and valid.txt
    with open(os.path.join(custom_path, 'train.txt'), 'w') as f:
        for image_name in training_image_names:
            f.write(f'data/custom/images/{image_name}\n')
    
    with open(os.path.join(custom_path, 'valid.txt'), 'w') as f:
        for image_name in validation_image_names:
            f.write(f'data/custom/images/{image_name}\n')


    # Copy and paste the label_dir to the custom folder under the name labels
    os.system(f'cp -r {args.labels_dir} {custom_path}/labels')

    # Copy and paste the data_folder to the custom folder under the name images
    os.system(f'cp -r {args.data_folder} {custom_path}/images')

    # traverse through the file names in the data_folder and if the corresponding label does not exist in the labels_dir, create an empty label txt file
    for image_name in tqdm(image_names):
        label_path = os.path.join(args.labels_dir, image_name.replace('.jpg', '.txt'))
        if not os.path.exists(label_path):
            with open(label_path, 'w') as f:
                pass 
        
        custom_label_path = os.path.join(custom_path, 'labels', image_name.replace('.jpg', '.txt'))
        if not os.path.exists(custom_label_path):
            with open(custom_label_path, 'w') as f:
                pass
    
    # count how many percent of txt files in the custom_label_path are empty
    empty_count = 0
    custom_label_path = os.path.join(custom_path, 'labels')
    for file in os.listdir(custom_label_path):
        if os.stat(os.path.join(custom_label_path, file)).st_size == 0:
            empty_count += 1
    
    print(f'User Warning: {round(empty_count/len(os.listdir(custom_label_path)) * 100, 2)}% of the txt files in the custom_label_path are empty, which means {round(empty_count/len(os.listdir(custom_label_path)) * 100, 2)}% of the regions dont contain annotations.')