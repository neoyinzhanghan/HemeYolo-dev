##############################################################################################
# IMPORTING PACKAGES
##############################################################################################

import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import argparse



##############################################################################################
# DEFINING FUNCTIONS
##############################################################################################

def get_region_dim(dim_dict, region_id):
    """
    Return the dimension tuple of region_id region if region_id in dim_dict, if not, find out, append to dim_dict and return it.
    """
    if region_id in dim_dict:
        return dim_dict[region_id]
    else:
        region_path = os.path.join(args.input_folder, region_id + '.jpg')
        img = Image.open(region_path)
        size = img.size

        dim_dict[region_id] = size

        return size

def is_valid_region_size(region_width, region_height, crop_width=512, crop_height=512):
    """ Return whether the region size is valid. """
    return crop_width<= region_width < crop_width*2 and crop_height <= region_height < crop_height*2

def get_quadrant(x, y, region_width, region_height, crop_width=512, crop_height=512):
    """ Return the quadrant in which (x,y) is located in the region. 
    --- output in ['TL', 'TR', 'BL', 'BR']
    """

    if not is_valid_region_size(region_width, region_height, crop_width=crop_width, crop_height=crop_height):
        raise ValueError(f'Invalid region size, both width and height must respectively be in [{crop_width},{crop_width*2}) and [{crop_height}, {crop_height*2}), your region size is {old_region_width}x{old_region_height}!')

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
 
def get_new_coordinate(x, y, region_width, region_height, quad, crop_width=512, crop_height=512):
    """ Return the correct quadrant coordinate. """

    if quad == 'TL':
        return x, y
    elif quad == 'TR':
        return x - (region_width - crop_width), y
    elif quad == 'BL':
        return x, y - (region_height - crop_height)
    elif quad == 'BR':
        return x - (region_width - crop_width), y - (region_height - crop_height)

def get_corners(x, y, radius, region_width, region_height):
    """ Return the top-left and bottom right corner of a square with radius centered at x, y, inside a region of specific width and height, the corners would be cutoff at the region boundary. """
    TL_corner = (max(x - radius, 0), max(y - radius, 0))
    BR_corner = (min(x + radius, region_width), min(y + radius, region_height))

    return TL_corner, BR_corner










##############################################################################################
# CALLING THE SCRIPT
##############################################################################################

if __name__ == '__main__':

    ##############################################################################################
    # ARGPARSE
    ##############################################################################################

    parser = argparse.ArgumentParser()

    ####################################
    group = parser.add_argument_group('Dataset and paths')
    ####################################

    group.add_argument('--input_folder', type=str,
                    help='Directory or directories containing input regions')
    group.add_argument('--file_path', type=str,
                    help='Path to your file containing labels and metadata')
    group.add_argument('--sheet_name', type=str, default=None,
                    help='If you are using .xlsx file (please dont) this is the name of the sheet containing your data')
    group.add_argument('--new_csv_path', type=str, 
                    help='Where would you like your new csv file to be saved, adapted to the cropped regions')

    ####################################
    group = parser.add_argument_group('Hyperparameters')
    ####################################

    group.add_argument('--radius', type=int, default=24,
                    help='If you want your patches to have size 48x48, for instance, you want radius to be 24')
    group.add_argument('--new_region_width', type=int, default=512,
                    help='Width of your new cropped regions')
    group.add_argument('--new_region_height', type=int, default=512,
                    help='Height of your new cropped regions')

    args = parser.parse_args()

    
    # Load the CSV file
    if os.path.splitext(args.file_path)[1] == '.csv':
        df = pd.read_csv(args.file_path)
        df = df.dropna(how='all')

    elif os.path.splitext(args.file_path)[1] == '.xlsx':
        df = pd.read_excel(args.file_path, sheet_name=args.sheet_name)
        df = df.dropna(how='all')
        print('Using xlsx file is deprecated, please convert the xlsx file to csv first, make sure it is formatted properly.')

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

        old_region_width, old_region_height = get_region_dim(dim_dict, old_region_id)
        box_width = args.radius * 2
        box_height = args.radius * 2

        quad = get_quadrant(old_center_x, 
                            old_center_y,
                            old_region_width, 
                            old_region_height, 
                            crop_width=args.new_region_width, 
                            crop_height=args.new_region_height)

        new_center_x, new_center_y = get_new_coordinate(old_center_x, 
                                                        old_center_y, 
                                                        old_region_width, 
                                                        old_region_height, 
                                                        quad, 
                                                        crop_width=args.new_region_width, 
                                                        crop_height=args.new_region_height)

        new_region_id = os.path.splitext(old_region_id)[0] + '_' + quad

        slide_sid = row['slide_sid']
        slide_name = row['slide_name']
        diagnoses = row['diagnoses']

        TL_corner, BR_corner = get_corners(new_center_x, new_center_y, args.radius, args.new_region_width, args.new_region_height)
        
        new_data = {'class': 0,
                    'cell_cid': cell_cid, 
                    'center_x': new_center_x, 
                    'center_y': new_center_y, 
                    'box_x_min': TL_corner[0],
                    'box_y_min': TL_corner[1],
                    'box_x_max': BR_corner[0],
                    'box_y_max': BR_corner[1],
                    'box_width': box_width,
                    'box_height': box_height,
                    'region_id': new_region_id,
                    'region_rid': region_rid,
                    'region_width': args.new_region_width,
                    'region_height': args.new_region_height,
                    'slide_sid': slide_sid,
                    'slide_name': slide_name,
                    'diagnoses': diagnoses}
        
        new_df.append(new_data)
        
    new_df = pd.DataFrame.from_records(new_df)
    new_df.to_csv(args.new_csv_path, index=False)