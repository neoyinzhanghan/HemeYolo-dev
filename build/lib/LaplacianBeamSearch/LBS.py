import numpy as np
from LaplacianBeamSearch.laplace import laplace_boundary
from PIL import ImageDraw
import PIL
import pandas as pd
import os
from tqdm import tqdm

def distance_to_nearest_white_pixel(dir_vec, base_pt, mask, cap=100):
    """ Mask is a 2d numpy array representing a picture, and the base_pt is a np.array (x,y) representing the starting point,
    and dir_vec is a np.array [x, y] representing the direction vector. This function returns the distance from the base_pt 
    to the farthest white pixel in the direction of dir_vec. The unit is the number of pixels. 
    If the distance ends up being greater than cap, return None. """

    # normalize the direction vector
    dir_vec = dir_vec / np.linalg.norm(dir_vec)

    # get the dimensions of the mask
    mask_height, mask_width = mask.shape

    # initialize the distance
    distance = 0

    # initialize the current point
    current_pt = base_pt

    # while the current point is still within the mask
    while 0 <= current_pt[0] < mask_height and 0 <= current_pt[1] < mask_width and mask[int(current_pt[1])][int(current_pt[0])] < 10: # the <10 is arbitrary

        # update the current point
        current_pt = current_pt + dir_vec

        # update the distance
        distance += 1

        # if the distance is greater than the cap, return None
        if distance > cap:
            return None

    return distance

def get_dir_vecs(density):
    """ Generate a list of direction vectors based on the density. 
    The directional vectors must be evenly distributed, and density is the number of vectors per 360 degrees. """

    # initialize the list of direction vectors
    dir_vecs = []

    # for each angle in the range of 0 to 360 degrees
    for angle in np.linspace(0, 2*np.pi, density, endpoint=False):

        # get the direction vector
        dir_vec = np.array([np.cos(angle), np.sin(angle)])

        # append the direction vector to the list
        dir_vecs.append(dir_vec)

    return dir_vecs

def get_distance_to_boundary(center, mask, core_radius=5, density=10, cap=100, lenience=0.1):
    """ The distance to boundary is computed as follow. 
    First, we generate a list of direction vectors based on the density. 
    Then, we compute the distance to the farthest white pixel in each direction, from the point center + core_radius * direction (normalized). 
    If the cap is hit and a white pixel is not found, we ignore the direction.
    We grab the bottom lenience*100% of the distances, and take the maximum of those distances."""

    # get the list of direction vectors
    dir_vecs = get_dir_vecs(density)

    # initialize the list of distances
    distances = []

    # for each direction vector
    for dir_vec in dir_vecs:

        # normalize the direction vector
        dir_vec = dir_vec / np.linalg.norm(dir_vec)

        # get the distance to the farthest white pixel
        distance = distance_to_nearest_white_pixel(dir_vec, center + core_radius * dir_vec, mask, cap=cap)

        # append the distance to the list
        if distance is not None:
            distances.append(distance)

    # if the list of distances is empty, return cap
    if len(distances) == 0:

        return cap

    # sort the distances
    distances.sort()

    # get the bottom lenience*100% of the distances
    bottom_distances = distances[:int(lenience * len(distances))]

    # if bottom_distances is empty, then increase lenience until it is not empty
    while bottom_distances == []:
        lenience = min(lenience + 1/density, 1)
        bottom_distances = distances[:int(lenience * len(distances))]

    # get the maximum distance
    min_distance = max(bottom_distances)

    return min_distance + core_radius

def get_box(center, mask, core_radius=5, density=10, cap=100, padding=5, lenience=0.1):
    """ Return the TL_x, TL_y, BR_x, BR_y of a box centered at center. 
    The box is the smallest box containing a circle of a radius that is equal to the distance to the boundary plus padding. """
    
    # get the distance to the boundary
    distance_to_boundary = get_distance_to_boundary(center, mask, core_radius=core_radius, density=density, cap=cap, lenience=lenience)

    # get the radius of the circle, call the ceiling function to make sure it is an integer
    radius = distance_to_boundary + padding
    radius = int(np.ceil(radius))

    # get the top left corner of the box
    TL_x = int(center[0] - radius)
    TL_y = int(center[1] - radius)

    # get the bottom right corner of the box
    BR_x = int(center[0] + radius)
    BR_y = int(center[1] + radius)

    # make sure the top left corner is within the image
    TL_x = max(TL_x, 0)
    TL_y = max(TL_y, 0)

    # make sure the bottom right corner is within the image
    BR_x = min(BR_x, mask.shape[0])
    BR_y = min(BR_y, mask.shape[1])

    return TL_x, TL_y, BR_x, BR_y, distance_to_boundary


def LBS_visualization(image_path, label_path, core_radius=3, density=32, cap=64, padding=15, lenience=0.1, dilation=3):
    """ This function takes the image_path, parse out each of the centers in the label file in the label_path,
    and then call the get_box function to get the TL_x, TL_y, BR_x, BR_y of the box for each center,
    and then use the TL_x, TL_y, BR_x, BR_y to annotate the image. """

    mask = laplace_boundary(image_path, prop_black=0.9, bins=128, dilation=3)
    # if the label_path is empty txt, skip it
    if os.stat(label_path).st_size == 0:
        raise ValueError('The label file is empty.')
    
    # open the label file as a pandas dataframe
    df = pd.read_csv(label_path, sep='\t', header=None)

    # rename the columns of df to class, center_x, center_y, width, height
    df = df.rename(columns={0: 'class', 1: 'center_x', 2: 'center_y', 3: 'width', 4: 'height'})

    # get the width and height of the image
    img = PIL.Image.open(image_path)
    img_width, img_height = img.size

    mask_pil = PIL.Image.fromarray(mask)

    # make mask_pil a RGB image
    mask_pil = mask_pil.convert('RGB')

    # traverse through the rows of the dataframe
    for index, row in tqdm(df.iterrows()):

        # get the row as a dictionary
        row_dict = row.to_dict()

        # get the center of the box
        center = np.array([row_dict['center_x'], row_dict['center_y']]) * np.array([img_width, img_height])

        # turn the center into an integer
        center = np.array([int(center[0]), int(center[1])])

        # call the get_box function
        TL_x, TL_y, BR_x, BR_y, distance = get_box(center, mask, core_radius=core_radius, density=density, cap=cap, padding=padding, lenience=lenience)

        # draw the box on the image in red using PIL and display it next to the mask
        img_draw = ImageDraw.Draw(img)
        img_draw.rectangle([TL_x, TL_y, BR_x, BR_y], outline='red')

        # draw a thick red dot at the center
        img_draw.ellipse([center[0] - core_radius, center[1] - core_radius, center[0] + core_radius, center[1] + core_radius], fill='red')

        # draw a thick red dot at the center on the mask_pil
        mask_pil_draw = ImageDraw.Draw(mask_pil)
        mask_pil_draw.ellipse([center[0] - core_radius, center[1] - core_radius, center[0] + core_radius, center[1] + core_radius], fill='red')

        # draw a circle with radius distance on the mask_pil
        mask_pil_draw.ellipse([center[0] - distance, center[1] - distance, center[0] + distance, center[1] + distance], outline='red')

        # draw the box on the mask_pil
        mask_pil_draw.rectangle([TL_x, TL_y, BR_x, BR_y], outline='red')


    # Put the annotated image and mask side by side
    img_concat = PIL.Image.new('RGB', (img.width + mask_pil.width, img.height))

    # paste the image on the left
    img_concat.paste(img, (0, 0))

    # paste the mask on the right
    img_concat.paste(mask_pil, (img.width, 0))

    # label the images
    img_concat_draw = ImageDraw.Draw(img_concat)
    img_concat_draw.text((0, 0), 'Original Image', fill='white')
    img_concat_draw.text((img.width, 0), 'Mask', fill='white')

    # return the concatenated image
    return img_concat




############################################################################
# SOME SCRIPT TO TEST THE FUNCTIONS
############################################################################

if __name__ == "__main__":

    image_dir = 'examples/images'
    label_dir = 'examples/labels'
    save_dir = 'examples/LBS_annotated'

    # get a list of image paths from the image_dir make sure file extension is jpg or png
    image_names = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir) if os.path.splitext(image_name)[1] in ['.jpg', '.png']]

    for image_name in tqdm(image_names):

        image_base_name = os.path.basename(image_name)
        image_ext = os.path.splitext(image_base_name)[1]

        # get the corresponding label path
        label_path = os.path.join(label_dir, image_base_name.replace(image_ext, '.txt'))

        img_concat = LBS_visualization(image_name, label_path, core_radius=7, density=64, cap=64, padding=20, lenience=0.1, dilation=3)

        # save the image
        img_concat.save(os.path.join(save_dir, image_base_name.replace(image_ext, '_annotated.png')))