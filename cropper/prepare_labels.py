##############################################################################################
# PACKAGE IMPORT
##############################################################################################

import os
import pandas as pd
import argparse
from tqdm import tqdm


#############################################################################################
# IMPORTANT UTILITY FUNCTION
#############################################################################################

def complete_region_ids(partial_region_ids):
    """Complete the region_id to contain all TL, TR, BL, BR regions. """
    
    quads = ['TL', 'TR', 'BL', 'BR']
    for region_id in partial_region_ids:
        root = region_id[:-2]
        for quad in quads:
            if root + quad not in partial_region_ids:
                partial_region_ids.append(root + quad)





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

    group.add_argument('--input_csv', type=str,
                    help='Directory or directories containing input csv')
    group.add_argument('--labels_dir', type=str,
                    help='Directory or directories to save output labels')

    args = parser.parse_args()

    pd.options.mode.chained_assignment = None  # Disable warning

    # Read the CSV file
    df = pd.read_csv(args.input_csv)

    # Create labels_dir if it doesn't exist
    if not os.path.exists(args.labels_dir):
        os.makedirs(args.labels_dir)

    # Sort by region_id
    df.sort_values('region_id', inplace=True)


    # Complete the region_id to contain all TL, TR, BL, BR regions.
    region_ids = list(df['region_id'].unique())
    complete_region_ids(region_ids)

    # Iterate over each unique region_id
    for region_id in tqdm(region_ids):
        if (df['region_id'] == region_id).any():
            # Filter dataframe by region_id
            region_df = df[df['region_id'] == region_id]

            region_df['center_x'] = region_df['center_x'] / region_df['region_width']
            region_df['center_y'] = region_df['center_y'] / region_df['region_height']
            region_df['box_width'] = region_df['box_width'] / region_df['region_width']
            region_df['box_height'] = region_df['box_height'] / region_df['region_height']
        
            # Select only the columns you want to kepp
            region_df = region_df[['class', 'center_x', 'center_y', 'box_width', 'box_height']]
            
            # Write to a new txt file, without header and index
            region_df.to_csv(f'{args.labels_dir}/{region_id}.txt', header=False, index=False, sep='\t')

        else:
            with open(f'{args.labels_dir}/{region_id}.txt', 'w') as f:
                pass
