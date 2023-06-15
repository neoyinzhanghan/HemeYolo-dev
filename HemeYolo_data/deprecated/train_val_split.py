import os
import random
from tqdm import tqdm
import shutil
import argparse

##############################################################################################
# SCRIPT
##############################################################################################

if __name__ == '__main__':

    ##############################################################################################
    # ARGPARSE
    ##############################################################################################

    parser = argparse.ArgumentParser()

    ####################################
    group = parser.add_argument_group('Dataset and paths')
    ####################################

    group.add_argument('--images_dir', type=str,
                        help='Directory or directories containing input images')
    group.add_argument('--train_dir', type=str,
                        help='Directory or directories to save training images')
    group.add_argument('--valid_dir', type=str,
                        help='Directory or directories to save validation images')
    group.add_argument('--train_prop', type=float, default=0.8,
                        help='The proportion of data to be used for training')

    args = parser.parse_args()

    # if train_dir does not exist, create it
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    # if valid_dir does not exist, create it
    if not os.path.exists(args.valid_dir):
        os.makedirs(args.valid_dir)

    # copy train_prop of images from images_dir randomly to train_dir and the rest to valid_dir
    image_names = [image_name for image_name in os.listdir(args.images_dir) if image_name.endswith('.jpg') or image_name.endswith('.png') ]
    random.shuffle(image_names)
    training_image_names = image_names[:int(len(image_names) * args.train_prop)]
    validation_image_names = image_names[int(len(image_names) * args.train_prop):]

    # copy images to train_dir and valid_dir, using file extention .jpg
    for image_name in tqdm(training_image_names):
        shutil.copy(os.path.join(args.images_dir, image_name), os.path.join(args.train_dir, image_name))

        # change file extension to .jpg
        os.rename(os.path.join(args.train_dir, image_name), os.path.join(args.train_dir, image_name.replace('.png', '.jpg')))

    for image_name in tqdm(validation_image_names):
        shutil.copy(os.path.join(args.images_dir, image_name), os.path.join(args.valid_dir, image_name))

        # change file extension to .jpg
        os.rename(os.path.join(args.valid_dir, image_name), os.path.join(args.valid_dir, image_name.replace('.png', '.jpg')))