import argparse
import tqdm
import os
import numpy as np
from PIL import Image
from HemeYolo_data.utils import check_labels, enforce_image_extension
from HemeYolo_data.augmentations import RandomAugmentor

##############################################################################################
# ARGPARSE
##############################################################################################

parser = argparse.ArgumentParser()

####################################
group = parser.add_argument_group('Dataset and paths')
####################################

group.add_argument('--data_dir', type=str,
                    help='Directory containing input images and their corresponding labels, assuming the following structure: data_dir/images and data_dir/labels')

args = parser.parse_args()


##############################################################################################
# THE SCRIPT
##############################################################################################

if __name__ == '__main__':

    """ 
    Augment each image in the data_dir/images and save the augmented images in the data_dir/images, and their corresponding labels in the data_dir/labels. 
    Augment each image one pass, and then two pass, resulting in tripling the number of images.
    """

    # first make sure that the images all have corresponding labels
    check_labels(os.path.join(args.data_dir, 'images'), os.path.join(args.data_dir, 'labels'))

    # enforce image extension to be .jpg
    enforce_image_extension(os.path.join(args.data_dir, 'images'))

    output_images_dir, output_labels_dir = os.path.join(args.data_dir, 'images'), os.path.join(args.data_dir, 'labels')

    # list all images in the data_dir/images
    files = [file for file in os.listdir(os.path.join(args.data_dir, 'images')) if file.endswith('.jpg')]

    for file in tqdm.tqdm(files):
        # get the basename of the file
        basename = os.path.basename(file)
        
        # use the RandomAugmentor class to augment the image
        augmentor = RandomAugmentor(os.path.join(args.data_dir, 'images', file), os.path.join(args.data_dir, 'labels', os.path.splitext(basename)[0] + '.txt'))

        # augment the image one pass
        augmentor.augment()

        # save the augmented image and label
        augmentor.save_most_recent_image(output_images_dir)
        augmentor.save_most_recent_label_df(output_labels_dir)

        # save the original image and label
        augmentor.save_original_image(output_images_dir)
        augmentor.save_original_label_df(output_labels_dir)

        # clear the augmentor
        augmentor.clear_augmentations()

        # augment the image two pass
        augmentor.augment_n_times(2, continue_on_error=True)

        # save the augmented image and label
        augmentor.save_most_recent_image(output_images_dir)
        augmentor.save_most_recent_label_df(output_labels_dir)