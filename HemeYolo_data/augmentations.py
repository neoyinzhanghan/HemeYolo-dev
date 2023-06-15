import os
from HemeYolo_data.utils import get_label_as_df
import cv2
import random
import numpy as np
from tqdm import tqdm
from HemeYolo_visual.annotate_regions import annotate_image
from PIL import Image, ImageDraw

""" All augmentation functions are defined using the following template. 
The inputs include the image to be augmented, the label df, and name. 
A new image, label df, and new name are returned.
Names are updated by appending the augmentation code to the original name separated by an underscore. """

""" Supported image augmentations do not change the dimension of the image or geometrically transform the image.
The following augmentations are supported:
    - Geometric transformations
        - HFLip
        - VFlip
        - Rotations (90, 180, 270 degrees)
        - CropNResize
        - Cutout
        - Blendout
    - Color transformations
        - Contrast
        - Saturation
        - Hue
        - ColorJitterI want you
        - Brightness
        - Blurr
        - Sharpen
        - Noise """

""" The label dfs are assumed to be in the following format: [class, center_x, center_y, box_width, box_height] in relative coordinates. """

####################################################################################################
# CLASS DEFINITION
####################################################################################################

# The replacement image path is named clot.jpg and is located in the exact same folder as this python file
# Make sure it works on all devices by using os.path.dirname(__file__)
replacement_image_path = os.path.join(os.path.dirname(__file__), 'clot.jpg')

class DeepHemeAugmentor():
    """ DeepHemeAugmentor class that contains all the augmentation methods, an original image, and original label df, and their augmentation sequences.

    === Attributes ===
    image: the original image
    label_df: the original label df
    image_sequence: a dictionary of augmented images
    label_sequence: a dictionary of augmented label dfs
    augmentation_sequence: a list containing the sequence of augmentations
    width: the width of the original image
    height: the height of the original image

    """

    def __init__(self, image_path, label_path) -> None:
        """ Initialize the Augmentor class with an image and a label df. """

        # open the image in cv2
        self.image = cv2.imread(image_path)

        # get the label df
        self.label_df = get_label_as_df(label_path)

        self.image_sequence = {'original': self.image}
        self.label_sequence = {'original': self.label_df}

        self.augmentation_sequence = ['original']

        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

    def get_augmentations(self):
        """ Return the augmentation sequence. """

        return self.augmentation_sequence
    

    ####################################################################################################
    # GEOMETRIC TRANSFORMATIONS
    ####################################################################################################

    def HFlip(self):
        """ Horizontal flip the image and label df. """

        # flip the image horizontally
        new_image = cv2.flip(self.image, 1)

        # flip the label df horizontally
        new_label_df = self.label_df.copy()
        new_label_df['center_x'] = 1 - new_label_df['center_x']

        # append the new image and label df to the sequence
        self.image_sequence['HFlip'] = new_image
        self.label_sequence['HFlip'] = new_label_df

        self.augmentation_sequence.append('HFlip')
    
    def VFlip(self):
        """ Vertical flip the image and label df. """
            
        # flip the image vertically
        new_image = cv2.flip(self.image, 0)

        # flip the label df vertically
        new_label_df = self.label_df.copy()
        new_label_df['center_y'] = 1 - new_label_df['center_y']

        # append the new image and label df to the sequence
        self.image_sequence['VFlip'] = new_image
        self.label_sequence['VFlip'] = new_label_df

        self.augmentation_sequence.append('VFlip')

    def Rot90(self):
        """ Rotate the image and label df 90 degrees clockwise. """

        # rotate the image 90 degrees clockwise
        new_image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)

        # rotate the label df 90 degrees clockwise
        new_label_df = self.label_df.copy()
        new_label_df['center_x'], new_label_df['center_y'] = 1 - new_label_df['center_y'], new_label_df['center_x']

        # append the new image and label df to the sequence
        self.image_sequence['Rot90'] = new_image
        self.label_sequence['Rot90'] = new_label_df

        self.augmentation_sequence.append('Rot90')
    
    def Rot180(self):
        """ Rotate the image and label df 180 degrees clockwise. """

        # rotate the image 180 degrees clockwise
        new_image = cv2.rotate(self.image, cv2.ROTATE_180)

        # rotate the label df 180 degrees clockwise
        new_label_df = self.label_df.copy()
        new_label_df['center_x'], new_label_df['center_y'] = 1 - new_label_df['center_x'], 1 - new_label_df['center_y']

        # append the new image and label df to the sequence
        self.image_sequence['Rot180'] = new_image
        self.label_sequence['Rot180'] = new_label_df

        self.augmentation_sequence.append('Rot180')
    
    def Rot270(self):
        """ Rotate the image and label df 270 degrees clockwise. """

        # rotate the image 270 degrees clockwise
        new_image = cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # rotate the label df 270 degrees clockwise
        new_label_df = self.label_df.copy()
        new_label_df['center_x'], new_label_df['center_y'] = new_label_df['center_y'], 1 - new_label_df['center_x']

        # append the new image and label df to the sequence
        self.image_sequence['Rot270'] = new_image
        self.label_sequence['Rot270'] = new_label_df

        self.augmentation_sequence.append('Rot270')

    def CropNResize(self, TL_x, TL_y, BR_x, BR_y):
        """ Crop the image and label df and resize it to the original size. 
        Preconditions: TL_x, TL_y, BR_x, BR_y are integers and TL_x < BR_x and TL_y < BR_y and BR_x < self.width and BR_y < self.height.""" 

        # check preconditions, raise ValueError if not met
        if not isinstance(TL_x, int) or not isinstance(TL_y, int) or not isinstance(BR_x, int) or not isinstance(BR_y, int):
            raise ValueError('TL_x, TL_y, BR_x, BR_y must be integers.')
        if TL_x >= BR_x or TL_y >= BR_y:
            raise ValueError('TL_x must be less than BR_x and TL_y must be less than BR_y.')
        if BR_x >= self.width or BR_y >= self.height:
            raise ValueError('BR_x must be less than self.width and BR_y must be less than self.height.')

        # crop the image
        new_image = self.image[TL_y:BR_y, TL_x:BR_x]

        # augmentation name is CropNResize_{TL_x}_{TL_y}_{BR_x}_{BR_y}
        augmentation_name = f'CropNResize_{TL_x}_{TL_y}_{BR_x}_{BR_y}'

        # copy the label df
        new_label_df = self.label_df.copy()

        # only keep the rows where center_x and center_y are between TL_x and BR_x and TL_y and BR_y respectively, note that the coordinates are relative to the original image
        new_label_df = new_label_df[(new_label_df['center_x'] >= TL_x / self.width) & (new_label_df['center_x'] <= BR_x / self.width) & (new_label_df['center_y'] >= TL_y / self.height) & (new_label_df['center_y'] <= BR_y / self.height)]

        # get the width and height of the cropped image
        new_label_df['center_x'] = (new_label_df['center_x'] - TL_x / self.width) / (BR_x / self.width - TL_x / self.width)
        new_label_df['center_y'] = (new_label_df['center_y'] - TL_y / self.height) / (BR_y / self.height - TL_y / self.height)

        # rescale the width and height of the cropped image
        new_label_df['box_width'] = new_label_df['box_width'] / ((BR_x - TL_x) / self.width)
        new_label_df['box_height'] = new_label_df['box_height'] / ((BR_y - TL_y) / self.height)

        # resize the image to the original size
        new_image = cv2.resize(new_image, (self.width, self.height))

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

    def _check_Cutout_Blendout_eligibiity(self):
        """ Check whether if Cutout or Blendout has never been applied before to the image, raise ValueError if so. 
        Private method. """

        for aug in self.augmentation_sequence:
            if 'Cutout' in aug or 'Blendout' in aug:
                return False

        return True

    def Cutout(self, TL_x, TL_y, BR_x, BR_y, replacement_image_path=replacement_image_path):
        """ Cutout a portion of the image and label df, and replace it with a rectangle of same size cut out of a different image, with slight blending. 
        Preconditions: If Cutout or Blendout has already been applied before to the image, raise ValueError."""

        # check preconditions, raise ValueError if not met
        if not self._check_Cutout_Blendout_eligibiity():
            raise ValueError('Cutout or Blendout has already been applied to the image.')
            
        # check preconditions, raise ValueError if not met
        if not isinstance(TL_x, int) or not isinstance(TL_y, int) or not isinstance(BR_x, int) or not isinstance(BR_y, int):
            raise ValueError('TL_x, TL_y, BR_x, BR_y must be integers.')
        if TL_x >= BR_x or TL_y >= BR_y:
            raise ValueError('TL_x must be less than BR_x and TL_y must be less than BR_y.')
        if BR_x >= self.width or BR_y >= self.height:
            raise ValueError('BR_x must be less than self.width and BR_y must be less than self.height.')

        # crop the image
        new_image = self.image.copy()

        # replace the cutout portion with the replacement image, make sure to blend the edges
        # It is as if the replacement image is translucent and pasted on top of the original image
        cover_opacity = random.uniform(0.6, 0.9) # this is the opacity of the replacement image
        new_image[TL_y:BR_y, TL_x:BR_x] = cv2.addWeighted(new_image[TL_y:BR_y, TL_x:BR_x], 1 - cover_opacity, cv2.imread(replacement_image_path)[TL_y:BR_y, TL_x:BR_x], cover_opacity, 0)

        # augmentation name is Cutout_{TL_x}_{TL_y}_{BR_x}_{BR_y}
        augmentation_name = f'Cutout_{TL_x}_{TL_y}_{BR_x}_{BR_y}'

        # copy the label df
        new_label_df = self.label_df.copy()

        # only keep the rows where center_x and center_y are between TL_x and BR_x and TL_y and BR_y respectively, note that the coordinates are relative to the original image
        new_label_df = new_label_df[(new_label_df['center_x'] < TL_x / self.width) | (new_label_df['center_x'] > BR_x / self.width) | (new_label_df['center_y'] < TL_y / self.height) | (new_label_df['center_y'] > BR_y / self.height)]

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)
    
    def Blendout(self, TL_x, TL_y, BR_x, BR_y, replacement_image_path=replacement_image_path):
        """ Cutout a portion of the image and label df, and replace it with a rectangle of same size cut out of a different image, with slight blending. 
        Preconditions: If Cutout or Blendout has already been applied before to the image, raise ValueError."""

        # check preconditions, raise ValueError if not met
        if not self._check_Cutout_Blendout_eligibiity():
            raise ValueError('Cutout or Blendout has already been applied to the image.')
            
        # check preconditions, raise ValueError if not met
        if not isinstance(TL_x, int) or not isinstance(TL_y, int) or not isinstance(BR_x, int) or not isinstance(BR_y, int):
            raise ValueError('TL_x, TL_y, BR_x, BR_y must be integers.')
        if TL_x >= BR_x or TL_y >= BR_y:
            raise ValueError('TL_x must be less than BR_x and TL_y must be less than BR_y.')
        if BR_x >= self.width or BR_y >= self.height:
            raise ValueError('BR_x must be less than self.width and BR_y must be less than self.height.')

        # crop the image
        new_image = self.image.copy()

        # replace the cutout portion with the replacement image, make sure to blend the edges
        # It is as if the replacement image is translucent and pasted on top of the original image
        cover_opacity = random.uniform(0.1, 0.4) # this is the opacity of the replacement image
        new_image[TL_y:BR_y, TL_x:BR_x] = cv2.addWeighted(new_image[TL_y:BR_y, TL_x:BR_x], 1 - cover_opacity, cv2.imread(replacement_image_path)[TL_y:BR_y, TL_x:BR_x], cover_opacity, 0)

        # augmentation name is Cutout_{TL_x}_{TL_y}_{BR_x}_{BR_y}
        augmentation_name = f'Cutout_{TL_x}_{TL_y}_{BR_x}_{BR_y}'

        # copy the label df, for blend out nothing is removed
        new_label_df = self.label_df.copy()

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

    
    ####################################################################################################
    # COLOR TRANSFORMATIONS
    ####################################################################################################

    def Contrast(self, alpha):
        """ Change the contrast of the image and label df.
        The alpha value for contrast has the following meaning:
            - alpha < 1: decrease
            - alpha = 1: no change
            - alpha > 1: increase

        Preconditions: alpha is a float and alpha > 0. 
        """

        alpha = float(alpha)
        # check preconditions, raise ValueError if not met
        if not isinstance(alpha, float):
            raise ValueError('alpha must be a float.')
        if alpha <= 0:
            raise ValueError('alpha must be greater than 0.')

        # change the contrast of the image
        new_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=0)

        # augmentation name is Contrast_{alpha}
        augmentation_name = f'Contrast_{alpha}'

        # copy the label df
        new_label_df = self.label_df.copy()

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

    def Saturation(self, alpha):
        """ Change the saturation of the image and label df.
        The alpha value for saturation has the following meaning:
            - alpha < 1: decrease
            - alpha = 1: no change
            - alpha > 1: increase
        Preconditions: alpha is a float and alpha > 0. 
        """

        # check preconditions, raise ValueError if not met

        alpha = float(alpha)
        if not isinstance(alpha, float):
            raise ValueError('alpha must be a float.')
        if alpha <= 0:
            raise ValueError('alpha must be greater than 0.')

        # change the saturation of the image
        new_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        new_image = cv2.convertScaleAbs(new_image, alpha=alpha, beta=0)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)

        # augmentation name is Saturation_{alpha}
        augmentation_name = f'Saturation_{alpha}'

        # copy the label df
        new_label_df = self.label_df.copy()

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)
    
    def Hue(self, alpha):
        """ Change the hue of the image and label df.
        The alpha value for hue has the following meaning:
            - alpha < 1: decrease
            - alpha = 1: no change
            - alpha > 1: increase

        Preconditions: alpha is a float and alpha > 0.
        """

        alpha = float(alpha)
        # check preconditions, raise ValueError if not met
        if not isinstance(alpha, float):
            raise ValueError('alpha must be a float.')
        if alpha <= 0:
            raise ValueError('alpha must be greater than 0.')

        # change the hue of the image
        new_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        new_image[:, :, 0] = new_image[:, :, 0] * alpha
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)

        # augmentation name is Hue_{alpha}
        augmentation_name = f'Hue_{alpha}'

        # copy the label df
        new_label_df = self.label_df.copy()

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

    def ColorJitter(self, alpha):
        """ Change the contrast, saturation, and hue of the image and label df. 
        The alpha value for color jitter has the following meaning:
            - alpha < 1: decrease
            - alpha = 1: no change
            - alpha > 1: increase
        Preconditions: alpha is a float and alpha > 0.
        """

        alpha = float(alpha)
        # check preconditions, raise ValueError if not met
        if not isinstance(alpha, float):
            raise ValueError('alpha must be a float.')
        if alpha <= 0:
            raise ValueError('alpha must be greater than 0.')

        # change the contrast, saturation, and hue of the image
        new_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        new_image = cv2.convertScaleAbs(new_image, alpha=alpha, beta=0)
        new_image[:, :, 0] = new_image[:, :, 0] * alpha
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)

        # augmentation name is ColorJitter_{alpha}
        augmentation_name = f'ColorJitter_{alpha}'

        # copy the label df
        new_label_df = self.label_df.copy()

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)


    ####################################################################################################
    # NON-COLOR TRANSFORMATIONS
    ####################################################################################################

    def Brightness(self, alpha):
        """ Change the brightness of the image and label df.
        The alpha value for brightness has the following meaning:
            - alpha < 1: decrease
            - alpha = 1: no change
            - alpha > 1: increase
            
        Preconditions: alpha is a float and alpha > 0.
        """

        alpha = float(alpha)
        # check preconditions, raise ValueError if not met
        if not isinstance(alpha, float):
            raise ValueError('alpha must be a float.')
        if alpha <= 0:
            raise ValueError('alpha must be greater than 0.')

        # change the brightness of the image
        new_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=0)

        # augmentation name is Brightness_{alpha}
        augmentation_name = f'Brightness_{alpha}'

        # copy the label df
        new_label_df = self.label_df.copy()

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)
    
    def Blur(self, kernel_size):
        """ Blur the image and label df.
        Preconditions: kernel_size is an odd integer and kernel_size > 0.
        """

        # check preconditions, raise ValueError if not met
        if not isinstance(kernel_size, int):
            raise ValueError('kernel_size must be an integer.')
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be an odd integer.')
        if kernel_size <= 0:
            raise ValueError('kernel_size must be greater than 0.')

        # blur the image
        new_image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)

        # augmentation name is Blur_{kernel_size}
        augmentation_name = f'Blur_{kernel_size}'

        # copy the label df
        new_label_df = self.label_df.copy()

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)
    
    def Sharpen(self, kernel_size):
        """ Sharpen the image and label df.
        Preconditions: kernel_size is an odd integer and kernel_size > 0.
        """

        # check preconditions, raise ValueError if not met
        if not isinstance(kernel_size, int):
            raise ValueError('kernel_size must be an integer.')
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be an odd integer.')
        if kernel_size <= 0:
            raise ValueError('kernel_size must be greater than 0.')

        # sharpen the image
        new_image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        new_image = cv2.addWeighted(self.image, 1.5, new_image, -0.5, 0)

        # augmentation name is Sharpen_{kernel_size}
        augmentation_name = f'Sharpen_{kernel_size}'

        # copy the label df
        new_label_df = self.label_df.copy()

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

    def Noise(self, alpha):
        """ Add noise to the image and label df. 
        The alpha value for noise has the following meaning:
            - alpha < 1: decrease
            - alpha = 1: no change
            - alpha > 1: increase
        Preconditions: alpha is a float and alpha > 0.
        """

        alpha = float(alpha)
        # check preconditions, raise ValueError if not met
        if not isinstance(alpha, float):
            raise ValueError('alpha must be a float.')
        if alpha <= 0:
            raise ValueError('alpha must be greater than 0.')

        # add noise to the image
        new_image = self.image.copy()
        new_image = new_image + alpha * new_image.std() * np.random.random(new_image.shape)

        # augmentation name is Noise_{alpha}
        augmentation_name = f'Noise_{alpha}'

        # copy the label df
        new_label_df = self.label_df.copy()

        # clip the image to be between 0 and 255
        new_image = np.clip(new_image, 0, 255)

        # convert the image to uint8
        new_image = new_image.astype(np.uint8)

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df
        
        self.augmentation_sequence.append(augmentation_name)
    
    def show_most_recent(self, show_original=False):
        """ Show the most recent augmentation result. 
        Add the center_x, center_y to the image as a red dot of radius 5.
        Add the box centered at center_x, center_y with width box_width and height box_height to the image as a red rectangle.
        Then show the image, and wait until the user presses a key to close the image. 
        Label the augmentation sequence on the image separated by >>>
        """

        # Make a clone of the most recent image to annotate in place
        most_recent_aug = self.augmentation_sequence[-1]
        img = self.image_sequence[most_recent_aug].copy()
        label = self.label_sequence[most_recent_aug]

        # Use the PIL library and ImageDraw function, but first you need to convert the image to PIL format
        img = Image.fromarray(img)

        # Annotate the image
        annotate_image(img, label)

        # Add the augmentation sequence to img using PIL, make the words red
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), ' >>> '.join(self.augmentation_sequence), (0, 0, 255))

        if not show_original:
            # Now show the image using cv2, wait until the user presses a key to close the image
            cv2.imshow('Most Recent Augmentation', np.array(img))
            cv2.waitKey(0)
        else:
            # Place the original image and the most recent augmentation side by side
            original_img = self.image_sequence['original'].copy()
            original_img = Image.fromarray(original_img)
            annotate_image(original_img, self.label_sequence['original'])
            draw = ImageDraw.Draw(original_img)
            draw.text((0, 0), 'Original', (0, 0, 255))
            img = np.hstack((np.array(original_img), np.array(img)))
            cv2.imshow('Most Recent Augmentation', img)
            cv2.waitKey(0)



####################################################################################################
# TESTING SCRIPT
####################################################################################################

if __name__ == '__main__':
    image_path = '/Users/neo/Documents/Research/DeepHeme/HemeYolo-dev/HemeYolo_data/6106_TL.jpg'
    label_path =  '/Users/neo/Documents/Research/DeepHeme/HemeYolo-dev/HemeYolo_data/6106_TL.txt'

    augmentor = DeepHemeAugmentor(image_path, label_path)

    augmentor.CropNResize(100, 100, 400, 400)

    augmentor.show_most_recent(show_original=True)