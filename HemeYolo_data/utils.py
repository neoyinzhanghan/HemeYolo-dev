import os

def get_label_as_df(label_path, label_type):
    # TODO
    pass

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

    # check that each image has a corresponding label
    for image_name in image_names:
        if image_name not in label_names:
            return False

    return True

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