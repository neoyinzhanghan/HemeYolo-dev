import argparse
from HemeYolo_data.crop import cut_images_in_folder, _crop_csv, _complete_region_ids
from HemeYolo_data.utils import check_labels, enforce_image_extension
import os
import pandas as pd
from tqdm import tqdm
import LaplacianBeamSearch.laplace as laplace
import numpy as np
import LaplacianBeamSearch.LBS as LBS

##############################################################################################
# ARGPARSE
##############################################################################################

parser = argparse.ArgumentParser()

####################################
group = parser.add_argument_group('Dataset and paths')
####################################

group.add_argument('--images_dir', type=str,
                    help='Directory or directories containing input image')
group.add_argument('--label_csv', type=str,
                    help='Path to your file containing labels and metadata')
group.add_argument('--output_dir', type=str,
                    help='Directory or directories to save your cropped images and corresponding labels')



####################################
group = parser.add_argument_group('Cropping parameters')
####################################

group.add_argument('--crop_width', type=int, default=512,
                    help='Width of the cropped region')
group.add_argument('--crop_height', type=int, default=512,
                    help='Height of the cropped region')
group.add_argument('--TL_only', type=bool, default=False,
                   help="remove all except the TL")

####################################
group = parser.add_argument_group('Hyperparameters')
####################################

group.add_argument('--box_method', default='LBS',
                    help='Method to use to find the bounding box of the cell. Valid inputs are LBS (Laplacian Beam Search) and integers (fixed box radius).')

args = parser.parse_args()


##############################################################################################
# THE SCRIPT
##############################################################################################

if __name__ == '__main__':
    
    # crop the images in the images_dir and save them in a folder named images in the output_dir, create the output_dir if it does not exist, same for the images

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, 'images'))
        os.makedirs(os.path.join(args.output_dir, 'labels'))
    
    else:
        if not os.path.exists(os.path.join(args.output_dir, 'images')):
            os.makedirs(os.path.join(args.output_dir, 'images'))
        if not os.path.exists(os.path.join(args.output_dir, 'labels')):
            os.makedirs(os.path.join(args.output_dir, 'labels'))

    cut_images_in_folder(images_dir=args.images_dir, output_dir=os.path.join(args.output_dir, 'images'), crop_width=args.crop_width, crop_height=args.crop_height)


    # grab the pandas dataframe output from the _crop_csv function

    df = _crop_csv(images_dir=args.images_dir, csv_path=args.label_csv, crop_width=args.crop_width, crop_height=args.crop_height)

    # Sort by region_id
    df.sort_values('region_id', inplace=True)

    # Complete the region_id to contain all TL, TR, BL, BR regions. This means that if a region_id is not present in the csv, 
    # it will still be labeled but the label file will be empty. This is useful for training the model, which will raise an error otherwise.
    region_ids = list(df['region_id'].unique())
    _complete_region_ids(region_ids)

    # Iterate over each unique region_id
    for region_id in tqdm(region_ids):

        if args.box_method == 'LBS':
            # Obtain the laplacian mask
            image_path = os.path.join(args.output_dir, 'images', region_id + '.jpg')
            mask = laplace.laplace_boundary(image_path, prop_black=0.9, bins=128, dilation=3, verbose=False)

        if (df['region_id'] == region_id).any():
            # Filter dataframe by region_id
            region_df = df[df['region_id'] == region_id]

            # create a new empty data frame with columns class, center_x_rel, center_y_rel, box_width_rel, box_height_rel
            new_region_df = pd.DataFrame(columns=['class', 'center_x_rel', 'center_y_rel', 'box_width_rel', 'box_height_rel'])

            # traverse through each row of the region_df
            for index, row in region_df.iterrows():
                # make the row in dictionary format
                row_dict = row.to_dict()

                cls = row_dict['class']

                # make sure to convert things to relative coordinates and create 
                center_x_rel = row_dict['center_x'] / row_dict['region_width']
                center_y_rel = row_dict['center_y'] / row_dict['region_height']

                try:
                    box_method = int(args.box_method)
                except ValueError:
                    box_method = args.box_method

                if type(box_method) is int:
                    box_width_rel = box_method*2 / row_dict['region_width']
                    box_height_rel = box_method*2 / row_dict['region_height']

                elif box_method == 'LBS':
                    # Grab the center as a numpy array
                    center = np.array([row_dict['center_x'], row_dict['center_y']])
                    _, _, _, _, distance_to_boundary = LBS.get_box(center, mask, core_radius=7, density=64, cap=64, padding=25, lenience=0.1)
                    radius = distance_to_boundary + 20
                    box_width_rel = radius*2 / row_dict['region_width']
                    box_height_rel = radius*2 / row_dict['region_height']
                
                else:
                    raise ValueError(f'Invalid box_method {args.box_method}')
        
                # Append the class, center_x_rel, center_y_rel, box_width_rel, box_height_rel to the new_region_df as a new row
                new_df_row = pd.DataFrame({'class': [cls], 
                                           'center_x_rel': [center_x_rel], 
                                           'center_y_rel': [center_y_rel], 
                                           'box_width_rel': [box_width_rel], 
                                           'box_height_rel': [box_height_rel]})
                
                new_region_df = pd.concat([new_region_df, new_df_row], ignore_index=True)
                
            # save the new_region_df as a txt file in the output_dir/labels, with the name {region_id}.txt, no header, no index
            new_region_df.to_csv(f'{args.output_dir}/labels/{region_id}.txt', header=False, index=False, sep='\t')

        else:
            # write an empty file named {region_id}.txt in the output_dir/labels
            with open(f'{args.output_dir}/labels/{region_id}.txt', 'w') as f:
                # no content in the file but make sure the file exists
                pass
        
    # first make sure that the images all have corresponding labels
    check_labels(os.path.join(args.output_dir, 'images'), os.path.join(args.output_dir, 'labels'))
    enforce_image_extension(os.path.join(args.output_dir, 'images'))

    if args.TL_only:
        # remove all files except ones whose basename end with TL in os.path.join(args.output_dir, 'images') and os.path.join(args.output_dir, 'labels')
        for file in os.listdir(os.path.join(args.output_dir, 'images')):
            if not file.endswith('TL.jpg'):
                os.remove(os.path.join(args.output_dir, 'images', file))
                os.remove(os.path.join(args.output_dir, 'labels', file.replace('.jpg', '.txt')))