import argparse
import os
from tqdm import tqdm
from HemeYolo_data.utils import check_labels, enforce_image_extension
import numpy as np


##############################################################################################
# ARGPARSE
##############################################################################################

parser = argparse.ArgumentParser()

####################################
group = parser.add_argument_group('Dataset and paths')
####################################

group.add_argument('--data_dir', type=str,
                    help='Directory containing input images and their corresponding labels, assuming the following structure: data_dir/images and data_dir/labels')
group.add_argument('--output_dir', type=str,
                    help='Directory to save your train, valid, test splits images and corresponding labels, assuming the following structure: output_dir/train/images, output_dir/train/labels, output_dir/valid/images, output_dir/valid/labels, output_dir/test/images, output_dir/test/labels')

####################################
group = parser.add_argument_group('Splitting parameters')
####################################

group.add_argument('--train_ratio', type=float, default=0.7,
                    help='Ratio of the dataset to use for training')
group.add_argument('--valid_ratio', type=float, default=0.15,
                    help='Ratio of the dataset to use for validation')
group.add_argument('--test_ratio', type=float, default=0.15,
                    help='Ratio of the dataset to use for testing')

args = parser.parse_args()



##############################################################################################
# THE SCRIPT
##############################################################################################

if __name__ == '__main__':
    # first make sure that the images all have corresponding labels
    check_labels(os.path.join(args.data_dir, 'images'), os.path.join(args.data_dir, 'labels'))
    
    # enforce image extension to be .jpg
    enforce_image_extension(os.path.join(args.data_dir, 'images'))

    # check the integrity of the split, making sure they are between 0 and 1 and sum to 1
    if args.train_ratio < 0 or args.train_ratio > 1:
        raise ValueError('train_ratio must be between 0 and 1.')
    if args.valid_ratio < 0 or args.valid_ratio > 1:
        raise ValueError('valid_ratio must be between 0 and 1.')
    if args.test_ratio < 0 or args.test_ratio > 1:
        raise ValueError('test_ratio must be between 0 and 1.')
    if args.train_ratio + args.valid_ratio + args.test_ratio != 1:
        raise ValueError('train_ratio, valid_ratio and test_ratio must sum to 1.')

    # if the output_dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, 'train'))
        os.makedirs(os.path.join(args.output_dir, 'train', 'images'))
        os.makedirs(os.path.join(args.output_dir, 'train', 'labels'))
        os.makedirs(os.path.join(args.output_dir, 'valid'))
        os.makedirs(os.path.join(args.output_dir, 'valid', 'images'))
        os.makedirs(os.path.join(args.output_dir, 'valid', 'labels'))
        os.makedirs(os.path.join(args.output_dir, 'test'))
        os.makedirs(os.path.join(args.output_dir, 'test', 'images'))
        os.makedirs(os.path.join(args.output_dir, 'test', 'labels'))

    # if the output_dir does exist, check if the train, valid, test folders exist, if not create them
    else:
        if not os.path.exists(os.path.join(args.output_dir, 'train')):
            os.makedirs(os.path.join(args.output_dir, 'train'))
            os.makedirs(os.path.join(args.output_dir, 'train', 'images'))
            os.makedirs(os.path.join(args.output_dir, 'train', 'labels'))
        if not os.path.exists(os.path.join(args.output_dir, 'valid')):
            os.makedirs(os.path.join(args.output_dir, 'valid'))
            os.makedirs(os.path.join(args.output_dir, 'valid', 'images'))
            os.makedirs(os.path.join(args.output_dir, 'valid', 'labels'))
        if not os.path.exists(os.path.join(args.output_dir, 'test')):
            os.makedirs(os.path.join(args.output_dir, 'test'))
            os.makedirs(os.path.join(args.output_dir, 'test', 'images'))
            os.makedirs(os.path.join(args.output_dir, 'test', 'labels'))

    # get the list of images in the data_dir, that end with .jpg
    images = [os.path.basename(x) for x in os.listdir(os.path.join(args.data_dir, 'images')) if x.endswith('.jpg')]

    # shuffle the images
    np.random.shuffle(images)

    # get the number of images
    n_images = len(images)

    # get the number of images for each split
    n_train = int(n_images * args.train_ratio)
    n_valid = int(n_images * args.valid_ratio)
    n_test = int(n_images * args.test_ratio)

    # get the list of images for each split
    train_images = images[:n_train]
    valid_images = images[n_train:n_train+n_valid]
    test_images = images[n_train+n_valid:]

    # copy the images to the output_dir, name the tqdm progress bar as moving train/valid/test images
    for image in tqdm(train_images, desc='Moving train images'):
        os.system('cp ' + os.path.join(args.data_dir, 'images', image) + ' ' + os.path.join(args.output_dir, 'train', 'images'))
    for image in tqdm(valid_images, desc='Moving valid images'):
        os.system('cp ' + os.path.join(args.data_dir, 'images', image) + ' ' + os.path.join(args.output_dir, 'valid', 'images'))
    for image in tqdm(test_images, desc='Moving test images'):
        os.system('cp ' + os.path.join(args.data_dir, 'images', image) + ' ' + os.path.join(args.output_dir, 'test', 'images'))
    
    # copy the labels to the output_dir, name the tqdm progress bar as moving train/valid/test labels
    for image in tqdm(train_images, desc='Moving train labels'):
        os.system('cp ' + os.path.join(args.data_dir, 'labels', image.split('.')[0] + '.txt') + ' ' + os.path.join(args.output_dir, 'train', 'labels'))
    for image in tqdm(valid_images, desc='Moving valid labels'):
        os.system('cp ' + os.path.join(args.data_dir, 'labels', image.split('.')[0] + '.txt') + ' ' + os.path.join(args.output_dir, 'valid', 'labels'))
    for image in tqdm(test_images, desc='Moving test labels'):
        os.system('cp ' + os.path.join(args.data_dir, 'labels', image.split('.')[0] + '.txt') + ' ' + os.path.join(args.output_dir, 'test', 'labels'))

    # Create a custom.yaml file in the output_dir following these instruction
    """path:  (dataset directory path)
    train: (Complete path to dataset train folder)
    test: (Complete path to dataset test folder) 
    valid: (Complete path to dataset valid folder)

    nc: 1

    names: ['WBC']
    """

    with open(os.path.join(args.output_dir, 'custom.yaml'), 'w') as f:
        f.write('path: ' + args.output_dir + '\n')
        f.write('train: ' + os.path.join(args.output_dir, 'train') + '\n')
        f.write('test: ' + os.path.join(args.output_dir, 'test') + '\n')
        f.write('valid: ' + os.path.join(args.output_dir, 'valid') + '\n')
        f.write('nc: 1\n')
        f.write("names: ['WBC']\n")