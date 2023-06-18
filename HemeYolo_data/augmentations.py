import os
from HemeYolo_data.utils import get_label_as_df, renormalize_probs, check_same_name
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
        - ColorJitter
    - Non-color transformations
        - Brightness
        - Blur
        - Sharpen
        - Noise """

""" The label dfs are assumed to be in the following format: [class, center_x, center_y, box_width, box_height] in relative coordinates. """

####################################################################################################
# MAIN SUPER CLASS DEFINITION
####################################################################################################

# The replacement image path is named clot.jpg and is located in the exact same folder as this python file
# Make sure it works on all devices by using os.path.dirname(__file__)
replacement_image_path = "/home/dog/neo/HemeYolo-dev/HemeYolo_data/clot.jpg"

class DeepHemeAugmentor():
    """ DeepHemeAugmentor class that contains all the augmentation methods, an original image, and original label df, and their augmentation sequences.

    === Attributes ===
    image: the current image
    label_df: the current label df
    image_sequence: a dictionary of augmented images
    label_sequence: a dictionary of augmented label dfs
    augmentation_sequence: a list containing the sequence of augmentations
    width: the width of the original image
    height: the height of the original image
    supported_augmentations: [HFlip, VFlip, Rot90, Rot180, Rot270, CropNResize, Cutout, Blendout, Contrast, Saturation, Hue, ColorJitter, Brightness, Blur, Sharpen, Noise]] 
    name: the name of the image

    """

    def __init__(self, image_path, label_path) -> None:
        """ Initialize the Augmentor class with an image and a label df. """

        # check that image_path and label_path have the same name
        if not check_same_name(image_path, label_path):
            raise ValueError('image_path and label_path must have the same base pre-extension name.')
        else:
            self.name = os.path.basename(os.path.splitext(image_path)[0])

        # open the image in cv2
        self.image = cv2.imread(image_path)

        # get the label df
        self.label_df = get_label_as_df(label_path)

        self.image_sequence = {'original': self.image}
        self.label_sequence = {'original': self.label_df}

        self.augmentation_sequence = ['original']

        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        self.supported_augmentations = ['HFlip', 'VFlip', 'Rot90', 'Rot180', 'Rot270', 'CropNResize', 'Cutout', 'Blendout', 'Contrast', 'Saturation', 'Hue', 'ColorJitter', 'Brightness', 'Blur', 'Sharpen', 'Noise']

    ####################################################################################################
    # GEOMETRIC TRANSFORMATIONS
    ####################################################################################################

    def HFlip(self):
        """ Horizontal flip the image and label df. """

        # flip the image horizontally
        new_image = cv2.flip(self.image, 1)
        
        if self.is_labelled():
            new_label_df = self.label_df.copy()
            # flip the label df horizontally
            new_label_df['center_x'] = 1 - new_label_df['center_x']
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence['HFlip'] = new_image
        self.label_sequence['HFlip'] = new_label_df

        self.augmentation_sequence.append('HFlip')

        self.image, self.label_df = new_image, new_label_df
    
    def VFlip(self):
        """ Vertical flip the image and label df. """
            
        # flip the image vertically
        new_image = cv2.flip(self.image, 0)

        if self.is_labelled():
            # flip the label df vertically
            new_label_df = self.label_df.copy()
            new_label_df['center_y'] = 1 - new_label_df['center_y']
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence['VFlip'] = new_image
        self.label_sequence['VFlip'] = new_label_df

        self.augmentation_sequence.append('VFlip')

        self.image, self.label_df = new_image, new_label_df

    def Rot90(self):
        """ Rotate the image and label df 90 degrees clockwise. """

        # rotate the image 90 degrees clockwise
        new_image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)

        if self.is_labelled():
            new_label_df = self.label_df.copy()
            # rotate the label df 90 degrees clockwise
            new_label_df['center_x'], new_label_df['center_y'] = 1 - new_label_df['center_y'], new_label_df['center_x']
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence['Rot90'] = new_image
        self.label_sequence['Rot90'] = new_label_df

        self.augmentation_sequence.append('Rot90')

        self.image, self.label_df = new_image, new_label_df
    
    def Rot180(self):
        """ Rotate the image and label df 180 degrees clockwise. """

        # rotate the image 180 degrees clockwise
        new_image = cv2.rotate(self.image, cv2.ROTATE_180)

        if self.is_labelled():
            new_label_df = self.label_df.copy()
            # rotate the label df 180 degrees clockwise
            new_label_df['center_x'], new_label_df['center_y'] = 1 - new_label_df['center_x'], 1 - new_label_df['center_y']
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence['Rot180'] = new_image
        self.label_sequence['Rot180'] = new_label_df

        self.augmentation_sequence.append('Rot180')

        self.image, self.label_df = new_image, new_label_df
    
    def Rot270(self):
        """ Rotate the image and label df 270 degrees clockwise. """

        # rotate the image 270 degrees clockwise
        new_image = cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.is_labelled():
            new_label_df = self.label_df.copy()
            # rotate the label df 270 degrees clockwise
            new_label_df['center_x'], new_label_df['center_y'] = new_label_df['center_y'], 1 - new_label_df['center_x']
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence['Rot270'] = new_image
        self.label_sequence['Rot270'] = new_label_df

        self.augmentation_sequence.append('Rot270')

        self.image, self.label_df = new_image, new_label_df

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

        

        if self.is_labelled():
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
        else:
            new_label_df = None

        # resize the image to the original size
        new_image = cv2.resize(new_image, (self.width, self.height))

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

        self.image, self.label_df = new_image, new_label_df

    def Cutout(self, TL_x, TL_y, BR_x, BR_y, replacement_image_path=replacement_image_path):
        """ Cutout a portion of the image and label df, and replace it with a rectangle of same size cut out of a different image, with slight blending. 
        Preconditions: If Cutout or Blendout has already been applied before to the image, raise ValueError."""

        # check preconditions, raise ValueError if not met
        if not self._check_Cutout_Blendout_eligibility():
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

        if cv2.imread(replacement_image_path) is None:
            raise ValueError(f'replacement_image_path {replacement_image_path}is invalid.')
        
        new_image[TL_y:BR_y, TL_x:BR_x] = cv2.addWeighted(new_image[TL_y:BR_y, TL_x:BR_x], 1 - cover_opacity, cv2.imread(replacement_image_path)[TL_y:BR_y, TL_x:BR_x], cover_opacity, 0)

        # augmentation name is Cutout_{TL_x}_{TL_y}_{BR_x}_{BR_y}
        augmentation_name = f'Cutout_{TL_x}_{TL_y}_{BR_x}_{BR_y}'

        if self.is_labelled():
            # copy the label df
            new_label_df = self.label_df.copy()
            # only keep the rows where center_x and center_y are between TL_x and BR_x and TL_y and BR_y respectively, note that the coordinates are relative to the original image
            new_label_df = new_label_df[(new_label_df['center_x'] < TL_x / self.width) | (new_label_df['center_x'] > BR_x / self.width) | (new_label_df['center_y'] < TL_y / self.height) | (new_label_df['center_y'] > BR_y / self.height)]
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

        self.image, self.label_df = new_image, new_label_df
    
    def Blendout(self, TL_x, TL_y, BR_x, BR_y, replacement_image_path=replacement_image_path):
        """ Cutout a portion of the image and label df, and replace it with a rectangle of same size cut out of a different image, with slight blending. 
        Preconditions: If Cutout or Blendout has already been applied before to the image, raise ValueError."""

        # check preconditions, raise ValueError if not met
        if not self._check_Cutout_Blendout_eligibility():
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

        if cv2.imread(replacement_image_path) is None:
            raise ValueError(f'replacement_image_path {replacement_image_path}is invalid.')
        
        new_image[TL_y:BR_y, TL_x:BR_x] = cv2.addWeighted(new_image[TL_y:BR_y, TL_x:BR_x], 1 - cover_opacity, cv2.imread(replacement_image_path)[TL_y:BR_y, TL_x:BR_x], cover_opacity, 0)

        # augmentation name is Cutout_{TL_x}_{TL_y}_{BR_x}_{BR_y}
        augmentation_name = f'Cutout_{TL_x}_{TL_y}_{BR_x}_{BR_y}'

        # copy the label df, for blend out nothing is removed
        if self.is_labelled():
            new_label_df = self.label_df.copy()
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

        self.image, self.label_df = new_image, new_label_df
    
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
        if self.is_labelled():
            new_label_df = self.label_df.copy()
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

        self.image, self.label_df = new_image, new_label_df

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
        if self.is_labelled():
            new_label_df = self.label_df.copy()
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

        self.image, self.label_df = new_image, new_label_df
    
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
        if self.is_labelled():
            new_label_df = self.label_df.copy()
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

        self.image, self.label_df = new_image, new_label_df

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
        if self.is_labelled():
            new_label_df = self.label_df.copy()
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

        self.image, self.label_df = new_image, new_label_df


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
        if self.is_labelled():
            new_label_df = self.label_df.copy()
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

        self.image, self.label_df = new_image, new_label_df
    
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
        if self.is_labelled():
            new_label_df = self.label_df.copy()
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

        self.image, self.label_df = new_image, new_label_df
    
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
        if self.is_labelled():
            new_label_df = self.label_df.copy()
        else:
            new_label_df = None

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df

        self.augmentation_sequence.append(augmentation_name)

        self.image, self.label_df = new_image, new_label_df

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
        if self.is_labelled():
            new_label_df = self.label_df.copy()
        else:
            new_label_df = None

        # clip the image to be between 0 and 255
        new_image = np.clip(new_image, 0, 255)

        # convert the image to uint8
        new_image = new_image.astype(np.uint8)

        # append the new image and label df to the sequence
        self.image_sequence[augmentation_name] = new_image
        self.label_sequence[augmentation_name] = new_label_df
        
        self.augmentation_sequence.append(augmentation_name)

        self.image, self.label_df = new_image, new_label_df


    ####################################################################################################
    # UTILITY METHODS
    ####################################################################################################

    def is_labelled(self):
        """ Return whether the label df is not None. """

        return self.label_df is not None
    
    def get_augmentations(self):
        """ Return the augmentation sequence. """

        return self.augmentation_sequence    


    def get_original_image(self):
        """ Get the original image. """
            
        return self.image_sequence['original']
    
    def get_original_label_df(self):
        """ Get the original label df. """

        return self.label_sequence['original']
    
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
            original_img = self.get_original_image().copy()
            original_img = Image.fromarray(original_img)
            annotate_image(original_img, self.get_original_label_df())
            draw = ImageDraw.Draw(original_img)
            draw.text((0, 0), 'Original', (0, 0, 255))
            img = np.hstack((np.array(original_img), np.array(img)))
            cv2.imshow('Most Recent Augmentation', img)
            cv2.waitKey(0)

    def save_most_recent_image(self, save_dir):
        """ Save the most recent augmentation result, as a .jpg file in save_dir. """
            
        # Make a clone of the most recent image to annotate in place
        most_recent_aug = self.augmentation_sequence[-1]

        # Convert the image from BGR to RGB
        rgb_img = self.image_sequence[most_recent_aug][..., ::-1]

        # Now create the PIL Image
        img = Image.fromarray(rgb_img)
        
        # Save the image
        tag = '_' + '>>>'.join(self.augmentation_sequence[1:])
        img.save(os.path.join(save_dir, self.name+f'{tag}.jpg'))

    def save_most_recent_label_df(self, save_dir):
        """ Save the most recent label df, as a \\t separated .txt file in save_dir. No header and index."""

        # Make a clone of the most recent label df to annotate in place
        most_recent_aug = self.augmentation_sequence[-1]

        if self.is_labelled():
            # Save the label df
            tag = '_' + '>>>'.join(self.augmentation_sequence[1:])
            self.label_sequence[most_recent_aug].to_csv(os.path.join(save_dir, f'{tag}.txt'), sep='\t', index=False, header=False)
        else:
            # Write an empty txt file
            tag = '_' + '>>>'.join(self.augmentation_sequence[1:])
            with open(os.path.join(save_dir, self.name+f'{tag}.txt'), 'w') as f:
                pass

    def save_original_image(self, save_dir):
        """ Save the original image, as a .jpg file in save_dir. """
            
        # Convert the image from BGR to RGB
        rgb_img = self.image_sequence['original'][..., ::-1]

        # Now create the PIL Image
        img = Image.fromarray(rgb_img)
        
        # Save the image
        tag = '_original'
        img.save(os.path.join(save_dir, self.name+f'{tag}.jpg'))

    def save_original_label_df(self, save_dir):
        """ Save the original label df, as a \\t separated .txt file in save_dir. No augmentation sequence is added. No header and index."""

        # Make a clone of the most recent label df to annotate in place
        label = self.label_sequence['original']

        if self.is_labelled():
            # Save the label df
            tag = '_original'
            label.to_csv(os.path.join(save_dir, f'{tag}.txt'), sep='\t', index=False, header=False)
        else:
            # Write an empty txt file
            tag = '_original'
            with open(os.path.join(save_dir, self.name+f'{tag}.txt'), 'w') as f:
                pass

    def clear_augmentations(self):
        """ Clear the augmentation sequence. """

        self.augmentation_sequence = self.augmentation_sequence[:1]
        self.image, self.label_df = self.image_sequence['original'], self.label_sequence['original']


    ####################################################################################################
    # CHECK PAST AUGMENTATIONS
    ####################################################################################################

    def _check_augmentation_eligibility(self, augmentation_name):
        """ Check whether if augmentation_name has never been applied before to the image, raise ValueError if so. 
        Private method. """

        for aug in self.augmentation_sequence:
            if augmentation_name in aug:
                return False
        
        return True
    
    def _check_Cutout_Blendout_eligibility(self):
        """ Check whether if Cutout or Blendout has never been applied before to the image, raise ValueError if so. 
        Private method. """

        return self._check_augmentation_eligibility('Cutout') and self._check_augmentation_eligibility('Blendout')

    def _check_flip_eligibility(self):
        """ Check whether if HFlip or VFlip has never been applied before to the image, raise ValueError if so. 
        Private method. """

        return self._check_augmentation_eligibility('HFlip') and self._check_augmentation_eligibility('VFlip')
    
    def _check_rotation_eligibility(self):
        """ Check whether if Rot90, Rot180, or Rot270 has never been applied before to the image, raise ValueError if so. 
        Private method. """

        return self._check_augmentation_eligibility('Rot90') and self._check_augmentation_eligibility('Rot180') and self._check_augmentation_eligibility('Rot270')








####################################################################################################
# AN AUGMENTOR SUB CLASS THAT PERFORMS A RANDOM AUGMENTATION ON THE IMAGE AND LABEL DF
####################################################################################################



default_probs = [0.2/5, 0.2/5, 0.2/5, 0.2/5, 0.2/5, # HFlips, VFlips, Rot90, Rot180, Rot270
                 0.4, # CropNResize
                 0.2/2, 0.2/2, # Cutout, Blendout
                 0.1/4, 0.1/4, 0.1/4, 0.1/4, # Contrast, Saturation, Hue, ColorJitter
                 0.1/4, 0.1/4, 0.1/4, 0.1/4,] # Brightness, Blur, Sharpen, Noise

class RandomAugmentor(DeepHemeAugmentor):
    """ An augmentor that performs a random augmentation on the image and label df. 
    A subclass of DeepHemeAugmentor.
    Implements a random augmentation from the supported augmentation methods with randomized parameter. 

    === Attributes ===
    ... attributes from super class DeepHemeAugmentor ...
    min_crop_prop: the minimum proportion of the width and height the crop must have
    max_cutout_prop: the maximum proportion of the width and height that can be cutout
    min_cutout_prop: the minimum proportion of the width and height that can be cutout
    max_blendout_prop: the maximum proportion of the image that can be blendout
    max_contrast_alpha: the maximum alpha value for contrast
    min_contrast_alpha: the minimum alpha value for contrast
    max_saturation_alpha: the maximum alpha value for saturation
    min_saturation_alpha: the minimum alpha value for saturation
    max_hue_alpha: the maximum alpha value for hue
    min_hue_alpha: the minimum alpha value for hue
    max_color_jitter_alpha: the maximum alpha value for color jitter
    min_color_jitter_alpha: the minimum alpha value for color jitter
    max_brightness_alpha: the maximum alpha value for brightness
    min_brightness_alpha: the minimum alpha value for brightness
    max_blur_kernel_size: the maximum kernel size for blurring
    min_blur_kernel_size: the minimum kernel size for blurring
    max_sharpen_kernel_size: the maximum kernel size for sharpening
    min_sharpen_kernel_size: the minimum kernel size for sharpening
    max_noise_alpha: the maximum alpha value for noise
    min_noise_alpha: the minimum alpha value for noise

    """

    def __init__(self, image_path, label_path,
                 max_crop_prop=0.8,
                 min_crop_prop=0.5,
                 max_cutout_prop=0.4,
                 min_cutout_prop=0.2,
                 max_blendout_prop=0.5,
                 min_blendout_prop=0.3,
                 max_contrast_alpha=1.3,
                 min_contrast_alpha=0.7,
                 max_saturation_alpha=1.3,
                 min_saturation_alpha=0.7,
                 max_hue_alpha=1.3,
                 min_hue_alpha=0.7,
                 max_color_jitter_alpha=1.3,
                 min_color_jitter_alpha=0.7,
                 max_brightness_alpha=1.3,
                 min_brightness_alpha=0.7,
                 max_blur_kernel_size=25,
                 min_blur_kernel_size=5,
                 max_sharpen_kernel_size=299,
                 min_sharpen_kernel_size=99,
                 max_noise_alpha=2,
                 min_noise_alpha=1,
                 probs=default_probs) -> None:
        super().__init__(image_path, label_path)

        # set the attributes
        self.max_crop_prop = max_crop_prop
        self.min_crop_prop = min_crop_prop
        self.max_cutout_prop = max_cutout_prop
        self.min_cutout_prop = min_cutout_prop
        self.max_blendout_prop = max_blendout_prop
        self.min_blendout_prop = min_blendout_prop
        self.max_contrast_alpha = max_contrast_alpha
        self.min_contrast_alpha = min_contrast_alpha
        self.max_saturation_alpha = max_saturation_alpha
        self.min_saturation_alpha = min_saturation_alpha
        self.max_hue_alpha = max_hue_alpha
        self.min_hue_alpha = min_hue_alpha
        self.max_color_jitter_alpha = max_color_jitter_alpha
        self.min_color_jitter_alpha = min_color_jitter_alpha
        self.max_brightness_alpha = max_brightness_alpha
        self.min_brightness_alpha = min_brightness_alpha
        self.max_blur_kernel_size = max_blur_kernel_size
        self.min_blur_kernel_size = min_blur_kernel_size
        self.max_sharpen_kernel_size = max_sharpen_kernel_size
        self.min_sharpen_kernel_size = min_sharpen_kernel_size
        self.max_noise_alpha = max_noise_alpha
        self.min_noise_alpha = min_noise_alpha
        self.probs=probs

    def _random_box_with_min(self, min_prop):
        """ Return the TL_x, TL_y, BR_x, BR_y of a random box with minimum proportion of min_crop_prop, fitting inside self.image. """
            
        # get a random box with minimum proportion of min_crop_prop, fitting inside self.image
        TL_x = random.randint(0, int((1 - min_prop) * self.width))
        TL_y = random.randint(0, int((1 - min_prop) * self.height))
        BR_x = random.randint(TL_x + int(min_prop * self.width), self.width)
        BR_y = random.randint(TL_y + int(min_prop * self.height), self.height)

        return TL_x, TL_y, BR_x, BR_y
    
    def _random_box_with_min_max(self, min_prop, max_prop):
        """ Return the TL_x, TL_y, BR_x, BR_y of a random box with minimum proportion of min_crop_prop and maximum proportion of max_crop_prop, fitting inside self.image. """
            
        # get a random box with minimum proportion of min_crop_prop and maximum proportion of max_crop_prop, fitting inside self.image
        TL_x = random.randint(0, int((1 - max_prop) * self.width))
        TL_y = random.randint(0, int((1 - max_prop) * self.height))
        BR_x = random.randint(TL_x + int(min_prop * self.width), TL_x + int(max_prop * self.width))
        BR_y = random.randint(TL_y + int(min_prop * self.height), TL_y + int(max_prop * self.height))

        return TL_x, TL_y, BR_x, BR_y

    def augment(self):
        """ Take a random sample from the supported augmentations and perform the augmentation. 
        Do not allow the same augmentation to be performed twice """

        # get a random augmentation from the supported augmentations that has not been applied before
        unused_augmentations = [aug for aug in self.supported_augmentations if self._check_augmentation_eligibility(aug)]
        usused_indices = [i for i in range(len(self.supported_augmentations)) if self._check_augmentation_eligibility(self.supported_augmentations[i])]

        # renormalize the probabilities
        new_probs = renormalize_probs(self.probs, usused_indices)

        # weigh the random choice by the probabilities
        augmentation = random.choices(unused_augmentations, weights=new_probs)[0]

        if augmentation == 'HFlip':
            if not self._check_flip_eligibility():
                raise EligibilityError('HFlip has already been applied to the image.')
            self.HFlip()

        elif augmentation == 'VFlip':
            if not self._check_flip_eligibility():
                raise EligibilityError('VFlip has already been applied to the image.')
            self.VFlip()

        elif augmentation == 'Rot90':
            if not self._check_rotation_eligibility():
                raise EligibilityError('Rot90 has already been applied to the image.')
            self.Rot90()

        elif augmentation == 'Rot180':
            if not self._check_rotation_eligibility():
                raise EligibilityError('Rot180 has already been applied to the image.')
            self.Rot180()

        elif augmentation == 'Rot270':
            if not self._check_rotation_eligibility():
                raise EligibilityError('Rot270 has already been applied to the image.')
            self.Rot270()

        elif augmentation == 'CropNResize':
            if not (self._check_augmentation_eligibility('CropNResize') and self._check_augmentation_eligibility('Cutout') and self._check_augmentation_eligibility('Blendout')):
                raise EligibilityError('CropNResize not eligible.')
            TL_x, TL_y, BR_x, BR_y = self._random_box_with_min_max(min_prop=self.min_crop_prop, max_prop=self.max_crop_prop)
            self.CropNResize(TL_x, TL_y, BR_x, BR_y)

        elif augmentation == 'Cutout':
            if not (self._check_augmentation_eligibility('CropNResize') and self._check_augmentation_eligibility('Cutout') and self._check_augmentation_eligibility('Blendout')):
                raise EligibilityError('CropNResize not eligible.')
            TL_x, TL_y, BR_x, BR_y = self._random_box_with_min_max(min_prop=self.min_cutout_prop, max_prop=self.max_cutout_prop)
            self.Cutout(TL_x, TL_y, BR_x, BR_y)

        elif augmentation == 'Blendout':
            if not (self._check_augmentation_eligibility('CropNResize') and self._check_augmentation_eligibility('Cutout') and self._check_augmentation_eligibility('Blendout')):
                raise EligibilityError('CropNResize not eligible.')
            TL_x, TL_y, BR_x, BR_y = self._random_box_with_min_max(min_prop=self.min_blendout_prop, max_prop=self.max_blendout_prop)
            self.Blendout(TL_x, TL_y, BR_x, BR_y)

        elif augmentation == 'Contrast':
            if not self._check_augmentation_eligibility('Contrast'):
                raise EligibilityError('Contrast has already been applied to the image.')
            alpha = random.uniform(self.min_contrast_alpha, self.max_contrast_alpha)
            self.Contrast(alpha)

        elif augmentation == 'Saturation':
            if not self._check_augmentation_eligibility('Saturation'):
                raise EligibilityError('Saturation has already been applied to the image.')
            
            alpha = random.uniform(self.min_saturation_alpha, self.max_saturation_alpha)
            self.Saturation(alpha)

        elif augmentation == 'Hue':
            if not self._check_augmentation_eligibility('Hue'):
                raise EligibilityError('Hue has already been applied to the image.')
            alpha = random.uniform(self.min_hue_alpha, self.max_hue_alpha)
            self.Hue(alpha)

        elif augmentation == 'ColorJitter':
            if not self._check_augmentation_eligibility('ColorJitter'):
                raise EligibilityError('ColorJitter has already been applied to the image.')
            alpha = random.uniform(self.min_color_jitter_alpha, self.max_color_jitter_alpha)
            self.ColorJitter(alpha)

        elif augmentation == 'Brightness':
            if not self._check_augmentation_eligibility('Brightness'):
                raise EligibilityError('Brightness has already been applied to the image.')
            alpha = random.uniform(self.min_brightness_alpha, self.max_brightness_alpha)
            self.Brightness(alpha)

        elif augmentation == 'Blur':
            if not self._check_augmentation_eligibility('Blur') and self._check_augmentation_eligibility('Sharpen'):
                raise EligibilityError('Blur or Sharpen has already been applied to the image.')
            # get a random odd integer between min_blur_kernel_size and max_blur_kernel_size
            kernel_size = random.randint(self.min_blur_kernel_size, self.max_blur_kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.Blur(kernel_size)

        elif augmentation == 'Sharpen':
            if not self._check_augmentation_eligibility('Blur') and self._check_augmentation_eligibility('Sharpen'):
                raise EligibilityError('Blur or Sharpen has already been applied to the image.')
            # get a random odd integer between min_sharpen_kernel_size and max_sharpen_kernel_size
            kernel_size = random.randint(self.min_sharpen_kernel_size, self.max_sharpen_kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.Sharpen(kernel_size)

        elif augmentation == 'Noise':
            if not self._check_augmentation_eligibility('Noise'):
                raise EligibilityError('Noise has already been applied to the image.')
            alpha = random.uniform(self.min_noise_alpha, self.max_noise_alpha)
            self.Noise(alpha)
        else:
            raise ValueError('Augmentation is not supported.')
    
    def augment_n_times(self, n, continue_on_error=False):
        """ Apply the augmentation function n times. 
        You might want to turn on continue_on_error, which may end up in infinite loops of EligibilityError
        """

        for i in range(n):
            try:
                self.augment()
            except EligibilityError:
                if continue_on_error:
                    continue
                else:
                    raise EligibilityError('Augmentation is not eligible. You are applying the same augmentation multiple times. The augmentation sampler is random,'+
                                        ' so if you want the program to continue sampling, turn on continue_on_error=True.')

class EligibilityError(Exception):
    """ An exception raised when an augmentation is not eligible to be performed. """
    pass

####################################################################################################
# TESTING SCRIPT
####################################################################################################

if __name__ == '__main__':
    image_path = '/Users/neo/Documents/Research/DeepHeme/HemeYolo-dev/HemeYolo_data/6106_TL.jpg'
    label_path =  '/Users/neo/Documents/Research/DeepHeme/HemeYolo-dev/HemeYolo_data/6106_TL.txt'

    augmentor = RandomAugmentor(image_path, label_path)

    augmentor.augment()

    # while True: # continue on error in the case of a double cutout or blendout
    #     try:
    #         augmentor.augment()
    #         break
    #     except EligibilityError:
    #         print('retrying')
    #         continue

    # augmentor.CropNResize(0, 0, 100, 100)

    augmentor.show_most_recent(show_original=True)