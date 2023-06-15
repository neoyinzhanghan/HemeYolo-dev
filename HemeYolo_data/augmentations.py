""" All augmentation functions are defined using the following template. 
The inputs include the image to be augmented, the label df, and name. 
A new image, label df, and new name are returned.
Names are updated by appending the augmentation code to the original name separated by an underscore. """

""" Supported image augmentations do not change the dimension of the image or geometrically transform the image.
The following augmentations are supported followed by the augmentation code in parentheses:
    - Geometric transformations (0)
        - Horizontal flip (0h)
        - Vertical flip (0v)
        - Rotations (90, 180, 270 degrees) (0a, 0b, 0c)
    - Color transformations (1)
        - Contrast (1a)
        - Saturation (1b)
        - Hue (1c)
    - Non-Color transformations (2)
        - Brightness (2a)
        - Blurring (2b)
        - Sharpening (2c)
        - Noise (2d) """

""" The label dfs are assumed to be in the following format: TBD """