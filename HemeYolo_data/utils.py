import os
import pandas as pd

def get_label_as_df(label_path):
    """ Read the label file as a pandas dataframe. 
    We are assuming the columns are in the following order: class, center_x, center_y, width, height. 
    If the label file is empty, return None."""
    
    if os.stat(label_path).st_size == 0:
        return None
    
    # open the label file as a pandas dataframe
    df = pd.read_csv(label_path, sep='\t', header=None)

    # rename the columns of df to class, center_x, center_y, width, height
    df = df.rename(columns={0: 'class', 1: 'center_x', 2: 'center_y', 3: 'box_width', 4: 'box_height'})

    return df

def get_output_as_df(output_path):
    """ Read the output file as a pandas dataframe.
    We are assuming the columns are in the following order: TL_x, TL_y, BR_x, BR_y, confidence, class. 
    If the output file is empty, return None."""

    if os.stat(output_path).st_size == 0:
        return None

    # open the output file as a pandas dataframe
    df = pd.read_csv(output_path, sep='\t', header=None)

    # rename the columns of df to TL_x, TL_y, BR_x, BR_y, confidence, class
    df = df.rename(columns={0: 'TL_x', 1: 'TL_y', 2: 'BR_x', 3: 'BR_y', 4: 'confidence', 5: 'class'})

    return df

def visualize_confidence():
    # TODO
    pass

def check_labels(images_dir, labels_dir):
    """ Check that each image in images_dir has a corresponding label in labels_dir named {image_name}.txt """

    # get a list of image paths from the images_dir make sure file extension is jpg or png
    image_names = [os.path.join(images_dir, image_name) for image_name in os.listdir(images_dir) if os.path.splitext(image_name)[1] in ['.jpg', '.png']]
    image_names = [os.path.basename(image_name) for image_name in image_names]
    image_names = [os.path.splitext(image_name)[0] for image_name in image_names]

    # get a list of label paths from the labels_dir make sure file extension is txt
    label_names = [os.path.join(labels_dir, label_name) for label_name in os.listdir(labels_dir) if os.path.splitext(label_name)[1] == '.txt']
    label_names = [os.path.basename(label_name) for label_name in label_names]
    label_names = [os.path.splitext(label_name)[0] for label_name in label_names]

    missing = []

    # check that each image has a corresponding label
    for image_name in image_names:
        if image_name not in label_names:
            missing.append(image_name)

    # for each missing label, create an empty label file same name the image file with .txt extension
    for image_name in missing:
        with open(os.path.join(labels_dir, image_name + '.txt'), 'w') as f:
            pass


def enforce_image_extension(images_dir, extension='.jpg'):
    """ Go through each image in the image_dir with either extensions .jpg or .png and rename it to .jpg. """

    # get a list of image paths from the images_dir make sure file extension is jpg or png
    image_names = [os.path.join(images_dir, image_name) for image_name in os.listdir(images_dir) if os.path.splitext(image_name)[1] in ['.jpg', '.png']]

    # go through each image and rename it to .jpg
    for image_name in image_names:
        image_base_name = os.path.basename(image_name)
        image_ext = os.path.splitext(image_base_name)[1]

        # rename the image if it is not already a .jpg
        if image_ext != extension:
            # get the new image name by replacing the last four characters with .jpg
            new_name = image_base_name[:-4] + extension
            os.rename(image_name, new_name)

def renormalize_probs(probs, indices_to_keep):
    """ Take a probability list and renormalize it so that the sum of the probabilities of the indices_to_keep is 1. """

    # get the sum of the probabilities of the indices_to_keep
    sum_probs = sum([probs[index] for index in indices_to_keep])

    # renormalize the probabilities
    probs = [probs[i] / sum_probs for i in range(len(probs)) if i in indices_to_keep]

    return probs

def check_same_name(image_path, label_path):
    """ Check that the image_path and label_path have the same basename. """

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    label_name = os.path.splitext(os.path.basename(label_path))[0]

    return image_name == label_name