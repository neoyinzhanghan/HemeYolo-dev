import os
import argparse
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

#############################################################################################
# IMPORTANT UTILITY FUNCTION
#############################################################################################

# type could be 'center-width-height' or 'top-left-bottom-right'
def annotate_image(image, label_df, label_type='center-width-height', core_radius=3):
    """ Annotate an image with a label. The quantities in the label_df are normalized relative to the image width and height.
    Draw the center of the box as a red dot with radius equal to core_radius and the box as a red rectangle.
    The dot is only drawn if label_type is 'center-width-height'.
    Bewarned that the image is modified in place. 
    We are assuming the image is in BGR format."""

    # get the image width and height
    width, height = image.size

    # traverse over all rows in the label_df
    for row in list(label_df.iterrows()):

        # get the row as a dictionary
        row = row[1]

        if label_type == 'center-width-height':
            # get the TL and BR coordinates
            TL_x = int((row['center_x'] - row['box_width']/2)*width)
            TL_y = int((row['center_y'] - row['box_height']/2)*height)
            BR_x = int((row['center_x'] + row['box_width']/2)*width)
            BR_y = int((row['center_y'] + row['box_height']/2)*height)

            # Make sure the coordinates are within the image
            TL_x = max(TL_x, 0)
            TL_y = max(TL_y, 0)
            BR_x = min(BR_x, width)
            BR_y = min(BR_y, height)

        elif label_type == 'top-left-bottom-right':
            # if the row has >=5 keys, raise an error
            assert len(row.keys()) >= 6, f'row has less than 6 keys, please check the label file, do you mean center-width-height?'

            # get the TL and BR coordinates
            TL_x = int((row['TL_x']))
            TL_y = int((row['TL_y']))
            BR_x = int((row['BR_x']))
            BR_y = int((row['BR_y']))

        # draw the rectangle in the image based on relative coordinates
        # does the image draw object need to be closed? 
        # https://stackoverflow.com/questions/26649716/close-a-pil-draw-object

        draw = ImageDraw.Draw(image)
        draw.rectangle(((TL_x, TL_y), (BR_x, BR_y)), outline='blue') # need to swap red and blue if BGR

        if label_type == 'center-width-height':
            # draw a red dot at the center
            draw.ellipse((row['center_x']*width - core_radius, row['center_y']*height - core_radius, row['center_x']*width + core_radius, row['center_y']*height + core_radius), fill='blue')


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

    group.add_argument('--input_dir', type=str,
                        help='Directory or directories containing input images')
    group.add_argument('--labels_dir', type=str,
                        help='Directory or directories containing input labels')  
    group.add_argument('--output_dir', type=str,
                        help='Directory or directories to save output annotated images')  
    group.add_argument('--label_type', type=str, default='center-width-height',
                        help='Type of label. Could be center-width-height or top-left-bottom-right')
    group.add_argument('--image_ext', type=str, default='.jpg',
                        help='Extension of the images in the input_dir')

    args = parser.parse_args()

    # if args.output_dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #  list all images in the input_dir, only .jpg files
    files = [file for file in os.listdir(args.input_dir) if file.endswith(args.image_ext)]

    for file in tqdm(files):

        continuation = True
        
        # get the basename of the file
        basename = os.path.basename(file)
        
        # label is named as basename but with .txt extension
        label_path = os.path.join(args.labels_dir, os.path.splitext(basename)[0] + '.txt')

        # check if the label file is empty, if so, save the image as is and continue
        if os.stat(os.path.join(label_path)).st_size == 0:
            image = Image.open(os.path.join(args.input_dir, file))
            image.save(os.path.join(args.output_dir, basename))
            
            continuation = False

        if continuation:
            # read the label_file as a dataframe
            df = pd.read_csv(label_path, sep='\t', header=None)

            if args.label_type == 'center-width-height':

                # convert the df to contain only the columns we need including 'center_x', 'center_y', 'box_width', 'box_height'
                df = df[[0, 1, 2, 3, 4]]
                df.columns = ['class', 'center_x', 'center_y', 'box_width', 'box_height']


                # annotate the image
                image = Image.open(os.path.join(args.input_dir, file))
                annotate_image(image, df, label_type=args.label_type)

                # save the image into the output_dir
                image.save(os.path.join(args.output_dir, basename))


            elif args.label_type == 'top-left-bottom-right':

                # rename the columns as [[x1, y1, x2, y2, confidence, class]]
                df.columns = ['TL_x', 'TL_y', 'BR_x', 'BR_y', 'confidence', 'class']

                # annotate the image
                image = Image.open(os.path.join(args.input_dir, file))
                annotate_image(image, df, label_type=args.label_type)

                # save the image into the output_dir
                image.save(os.path.join(args.output_dir, basename))
