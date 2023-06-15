##############################################################################################
# PACKAGE IMPORT
##############################################################################################

import os
from PIL import Image
import glob
from tqdm import tqdm
import pandas as pd


##############################################################################################
# DEFINE FUNCTIONS
##############################################################################################


# CROPPING IMAGES

def cut_image_into_quadrants(img_path, quadrant_size=512):
    """ Takes an image path and returns a tuple of 4 images, each cropped to a quadrant of the original image.
    The quadrants are defined as the top-left, top-right, bottom-left, and bottom-right quadrants,
    with corners stuck into the original image.
    The quadrant size is defined by the quadrant_size parameter.
    """

    # Open image file
    img = Image.open(img_path)

    # Verify image dimensions ### no need as the code is adaptive to size
    if img.size[0] < quadrant_size or img.size[1] < quadrant_size:
        raise ValueError(f"Image width and height must both be at least {quadrant_size} pixels, your image is {img.size[0]}x{img.size[1]} pixels!")

    # Define box parameters (top-left, top-right, bottom-left, bottom-right)
    box_TL = (0, 0, quadrant_size, quadrant_size)
    box_TR = (img.size[0] - quadrant_size, 0, img.size[0], quadrant_size)
    box_BL = (0, img.size[1]-quadrant_size, quadrant_size, img.size[1])
    box_BR = (img.size[0] - quadrant_size, img.size[1]-quadrant_size, img.size[0], img.size[1])

    # Crop images and save
    img_TL = img.crop(box_TL)
    img_TR = img.crop(box_TR)
    img_BL = img.crop(box_BL)
    img_BR = img.crop(box_BR)

    # Return as tuple
    return (img_TL, img_TR, img_BL, img_BR)

def cut_images_in_folder(input_folder, output_folder, quadrant_size=512):
    """ Takes an input folder and an output folder and cuts all images in the input folder into quadrants,
    saving the quadrants in the output folder. The filenames are modified to include the quadrant code.
    The quadrant code is appended to the original filename separated by an underscore.
    The quadrant codes are as follows: 
        - Top-left: TL
        - Top-right: TR
        - Bottom-left: BL
        - Bottom-right: BR
    """

    # Get list of jpg files in the input folder
    jpg_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    # Iterate through each .jpg file in the input folder with a progress bar
    for img_path in tqdm(jpg_files, desc="Processing images"):
        # Extract the base filename without extension
        base_filename = os.path.basename(os.path.splitext(img_path)[0])

        # Apply the previous function
        img_TL, img_TR, img_BL, img_BR = cut_image_into_quadrants(img_path, quadrant_size=quadrant_size)

        # Save new images in the output folder with modified filenames
        img_TL.save(os.path.join(output_folder, f"{base_filename}_TL.jpg"))
        img_TR.save(os.path.join(output_folder, f"{base_filename}_TR.jpg"))
        img_BL.save(os.path.join(output_folder, f"{base_filename}_BL.jpg"))
        img_BR.save(os.path.join(output_folder, f"{base_filename}_BR.jpg"))


# CROPPING LABELS

def _get_region_dim(images_dir, dim_dict:dict, region_id:str):
    """
    Return the dimension tuple of region_id region if region_id in dim_dict, if not, find out, append to dim_dict and return it.

    Private function.
    """
    if region_id in dim_dict:
        return dim_dict[region_id]
    else:
        region_path = os.path.join(images_dir, region_id + '.jpg')
        img = Image.open(region_path)
        size = img.size

        dim_dict[region_id] = size

        return size
    

def is_valid_region_size(region_width:int, region_height:int, crop_width:int=512, crop_height:int=512) -> bool:
    """ Return whether the region size is valid. """

    return crop_width<= region_width < crop_width*2 and crop_height <= region_height < crop_height*2


def get_quadrant(x:int, y:int, region_width:int, region_height:int, crop_width:int=512, crop_height:int=512) -> str:
    """ Return the quadrant in which (x,y) is located in the region. 
    --- output in ['TL', 'TR', 'BL', 'BR']
    """

    if not is_valid_region_size(region_width, region_height, crop_width=crop_width, crop_height=crop_height):
        raise ValueError(f'Invalid region size, both width and height must respectively be in [{crop_width},{crop_width*2}) and [{crop_height}, {crop_height*2}), your region size is {region_width}x{region_height}!')

    if 0 <= x < crop_width and 0 <= y < crop_height:
        return 'TL'
    elif region_width - crop_width <= x < region_width and 0 <= y < crop_height:
        return 'TR'
    elif 0 <= x < crop_width and region_height - crop_height <= y < region_height:
        return 'BL'
    elif region_width - crop_width <= x < region_width and region_height - crop_height <= y < region_height:
        return 'BR'
    else:
        raise ValueError('There is something wrong with the coordinate values and region width and height such that no valid quadrant is computed!')    
 

def get_new_coordinate(x:int, y:int, region_width:int, region_height:int, quad:int, crop_width:int=512, crop_height:int=512) -> tuple:
    """ Return the correct quadrant coordinate. """

    if quad == 'TL':
        return x, y
    elif quad == 'TR':
        return x - (region_width - crop_width), y
    elif quad == 'BL':
        return x, y - (region_height - crop_height)
    elif quad == 'BR':
        return x - (region_width - crop_width), y - (region_height - crop_height)
    

def get_corners(x:int, y:int, box_width:int, box_height:int, region_width:int, region_height:int) -> tuple:
    """ Return the top-left and bottom right corner of a rectangle centered at x, y, with box_width and box_height inside a region of specific width and height, the corners would be cutoff at the region boundary. """
    TL_x, TL_y = max(x - box_width, 0), max(y - box_height, 0)
    BR_x, BR_y = min(x + box_width, region_width), min(y + box_height, region_height)

    return TL_x, TL_y, BR_x, BR_y


# the return type is pandas dataframe
def _crop_csv(images_dir:str, csv_path:str, crop_width:int=512, crop_height:int=512) -> pd.DataFrame:

    """ Use the images_dir to update the csv_path in order for this csv_path to correspond to the cropped images.
    The csv_path must lead to a .csv file.
    The csv_path is assumed to be corresponding to the images in the images_dir. 
    The csv file is assumed to be in the following format:
    cell_cid	center_x	center_y	region_id	region_rid	region_width	region_height	slide_sid	slide_name	diagnoses

    The output pandas dataframe is in the following format:
    class	cell_cid	center_x	center_y	region_id	region_rid	region_width	region_height	slide_sid	slide_name	diagnoses

    The output pandas dataframe would correspond to the cropped images in the images_dir, processed by cut_images_in_folder.

    Private function.
    """

    # Load the CSV file
    if os.path.splitext(csv_path)[1] == '.csv':
        df = pd.read_csv(csv_path)
        df = df.dropna(how='all')

    else:
        raise ValueError('The file path must lead to a .csv file!')
    # print(df.columns) # printing only for debugging purposes

    # This column is to remind developed of the dataframe columns for debugging purposes
    # columns = ['cell_cid', 'center_x', 'center_y', 'box_x_min', 'box_x_max', 'box_y_min', 'box_y_max', 'region_id', 'region_rid', 'region_width', 'region_height', 'slide_sid', 'slide_name', 'diagnoses']

    new_df = []

    dim_dict = {} # a running dictionary that stores the information about the dimension of a region 

    # Iterate over the dataframe rows, and create a new data frame for cropped regions
    for index, row in tqdm(df.iterrows(), desc="Processing CSV"):
        cell_cid = row['cell_cid']
        old_center_x, old_center_y = row['center_x'], row['center_y']
        old_region_id = str(row['region_id'])
        region_rid = str(row['region_rid'])

        old_region_width, old_region_height = _get_region_dim(images_dir, dim_dict, old_region_id)

        quad = get_quadrant(old_center_x, 
                            old_center_y,
                            old_region_width, 
                            old_region_height, 
                            crop_width=crop_width, 
                            crop_height=crop_height)

        new_center_x, new_center_y = get_new_coordinate(old_center_x, 
                                                        old_center_y, 
                                                        old_region_width, 
                                                        old_region_height, 
                                                        quad, 
                                                        crop_width=crop_width,
                                                        crop_height=crop_height)

        new_region_id = os.path.splitext(old_region_id)[0] + '_' + quad

        slide_sid = row['slide_sid']
        slide_name = row['slide_name']
        diagnoses = row['diagnoses']

        new_data = {'class': 0,
                    'cell_cid': cell_cid, 
                    'center_x': new_center_x, 
                    'center_y': new_center_y, 
                    'region_id': new_region_id,
                    'region_rid': region_rid,
                    'region_width': crop_height,
                    'region_height': crop_height,
                    'slide_sid': slide_sid,
                    'slide_name': slide_name,
                    'diagnoses': diagnoses}
        
        new_df.append(new_data)
        
    new_df = pd.DataFrame.from_records(new_df)

    return new_df

def _complete_region_ids(partial_region_ids:list) -> None:
    """ Modify the list of region_ids to contain all TL, TR, BL, BR regions. 

    Private function.
    """
    
    quads = ['TL', 'TR', 'BL', 'BR']
    for region_id in partial_region_ids:
        root = region_id[:-2]
        for quad in quads:
            if root + quad not in partial_region_ids:
                partial_region_ids.append(root + quad)