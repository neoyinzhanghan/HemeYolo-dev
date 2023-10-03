# Python program to  Edge detection 
# using OpenCV in Python
# using Sobel edge detection 
# and laplacian method
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import PIL
from PIL import ImageDraw
import sys


############################################################################
# FUNCTIONS
############################################################################

def pixel_ecdf(img):
    # Flatten the image into 1 dimension
    pixels = img.flatten()

    # Calculate histogram (proportion of pixels at each intensity level)
    hist, bin_edges = np.histogram(pixels, bins=128, range=(0,256), density=True)

    # Calculate cumulative sum to get empirical distribution function
    edf = np.cumsum(hist)

    # Generate the intensity level thresholds
    intensity_level_thresholds = np.linspace(0, 255, num=256)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.step(intensity_level_thresholds, edf, where='post')
    plt.title("Empirical Distribution Function of Pixel Intensities")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Proportion of Pixels <= Intensity Level")
    plt.grid()
    plt.show()

def pixel_histogram(img):
    # Flatten the image into 1 dimension
    pixels = img.flatten()

    # Calculate histogram (counts of pixels at each intensity level)
    counts, bin_edges = np.histogram(pixels, bins=128, range=(0,256))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], counts, width = 1)
    plt.title("Histogram of Pixel Intensities")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count of Pixels")
    plt.grid()
    plt.show()

def get_threshold(img, prop_black=0.9, bins=256):
    # Flatten the image into 1 dimension
    pixels = img.flatten()

    # Calculate histogram (counts of pixels at each intensity level)
    counts, bin_edges = np.histogram(pixels, bins=bins, range=(0,256))

    # Calculate the cumulative sum of the histogram in reverse order
    cumulative_counts = np.cumsum(counts[::-1])[::-1]

    # Calculate the difference between each consecutive bin
    diffs = np.diff(counts)

    # Adjust the diffs array so we only consider thresholds where the proportion of pixels above the threshold is less than 10%
    adjusted_diffs = np.where(cumulative_counts[:-1] > prop_black * len(pixels), -np.inf, diffs)

    # Find the index where the adjusted difference is maximum (biggest drop)
    threshold_idx = np.argmax(adjusted_diffs)

    # The corresponding intensity level is our threshold
    threshold = bin_edges[threshold_idx]

    return threshold


def laplace_boundary(image_path, prop_black= 0.9, bins=256, verbose=False, dilation=3):
    """Returns the boundary of the image, if verbose, display all intermediary images"""

    # get the image as a numpy array, 3 channels (RGB)
    img_orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if verbose:
        # display the image and pause the execution until the user presses a key
        cv2.imshow("Original", img_orig)
        cv2.waitKey(0)
        # close all windows
        cv2.destroyAllWindows()

    # apply pyramid mean shift filtering
    pmsf= cv2.pyrMeanShiftFiltering(img_orig, 21, 51)
    if verbose:
        try:
            # display the image and pause the execution until the user presses a key
            cv2.imshow("pyrMeanShiftFiltering", pmsf)
            cv2.waitKey(0)
            # close all windows
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            # end the program
            sys.exit()

    # denoise pmsf
    img = cv2.fastNlMeansDenoisingColored(pmsf, None, 10, 10, 7, 21)
    if verbose:
        try:
            # display the image and pause the execution until the user presses a key
            cv2.imshow("fastNlMeansDenoisingColored", img)
            cv2.waitKey(0)
            # close all windows
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            # end the program
            sys.exit()

    # convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if verbose:
        try:
            # display the image and pause the execution until the user presses a key
            cv2.imshow("greyscale", img)
            cv2.waitKey(0)
            # close all windows
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            # end the program
            sys.exit()

    # apply Laplacian edge detection to img
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.absolute(laplacian)  # Absolute value
    laplacian = np.uint8(255 * (laplacian / np.max(laplacian)))  # Normalize to 0-255
    if verbose:
        try:
        # display the image and pause the execution until the user presses a key
            cv2.imshow("Laplacian", laplacian)
            cv2.waitKey(0)
            # close all windows
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            # end the program
            sys.exit()

    # # thickens the prominent edges and thin out the less prominent ones with bilateral filtering
    # laplacian2 = cv2.bilateralFilter(laplacian, 9, 75, 75)
    # if verbose:
    #     try:
    #     # display the image and pause the execution until the user presses a key
    #         cv2.imshow("bilateralFilter", laplacian2)
    #         cv2.waitKey(0)
    #         # close all windows
    #         cv2.destroyAllWindows()
    #     except KeyboardInterrupt:
    #         print("KeyboardInterrupt")
    #         # end the program
    #         sys.exit()

    # Enhance edges using Dilation
    kernel = np.ones((dilation,dilation),np.uint8)
    laplacian3 = cv2.dilate(laplacian, kernel, iterations = 1)
    if verbose:
        try:
        # display the image and pause the execution until the user presses a key
            cv2.imshow("dilate", laplacian)
            cv2.waitKey(0)
            # close all windows
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            # end the program
            sys.exit()

    # # Enhance edges using Unsharp Mask
    # gaussian = cv2.GaussianBlur(laplacian, (9,9), 10.0)
    # laplacian4 = cv2.addWeighted(laplacian, 1.5, gaussian, -0.5, 0, laplacian)
    # if verbose:
    #     try:
    #         # display the image and pause the execution until the user presses a key
    #         cv2.imshow("Unsharp Mask", laplacian4)
    #         cv2.waitKey(0)
    #         # close all windows
    #         cv2.destroyAllWindows()
    #     except KeyboardInterrupt:
    #         print("KeyboardInterrupt")
    #         # end the program
    #         sys.exit()

    # # Create intense contrast
    # # Find the min and max pixel values
    # min_val, max_val = np.min(laplacian4), np.max(laplacian4)
    # # Stretch the contrast
    # img_stretched = ((laplacian4 - min_val) / (max_val - min_val)) * 255
    # if verbose:
    #     try:
    #     # display the image and pause the execution until the user presses a key
    #         cv2.imshow("Contrast", img_stretched)
    #         cv2.waitKey(0)
    #         # close all windows
    #         cv2.destroyAllWindows()
    #         # quit the enter program once the user keyboard interrupts
    #         # sys.exit()
    #     except KeyboardInterrupt:
    #         print("KeyboardInterrupt")
    #         # end the program
    #         sys.exit()


    # # Convert float image to int
    # img_stretched = img_stretched.astype(np.uint8)

    _, img_thresholded = cv2.threshold(laplacian3, get_threshold(laplacian3, prop_black=prop_black, bins=bins), 255, cv2.THRESH_BINARY)
    if verbose:
        try:
        # display the image and pause the execution until the user presses a key
            cv2.imshow("thresholded", img_thresholded)
            cv2.waitKey(0)
            # close all windows
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            # end the program
            sys.exit()

    return img_thresholded


############################################################################
# MAIN
############################################################################

if __name__ == "__main__":

    ###########################################################################
    # ARGUMENT PARSER
    ############################################################################

    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()


    ##########################
    group = parser.add_argument_group("Datapaths and directories")
    ##########################

    group.add_argument("--image_dir", type=str,
                        help="Path to the directory containing the images to be processed")

    group.add_argument("--output_dir", type=str,
                        help="Path to the directory where the processed images will be saved")


    ##########################
    group = parser.add_argument_group("Hyper Parameters")
    ##########################

    group.add_argument("--prop_black", type=float, default=0.9,
                        help="Black pixel proportion required for thresholding calculation")

    group.add_argument("--bin", type=int, default=128,
                        help="Number of bins to use in the histogram used for thresholding calculation")

    group.add_argument("--present_original", type=bool, default=True,
                        help="Whether to present the original image")
    
    group.add_argument("--verbose", type=bool, default=False,
                        help="Whether to present the intermediary images")

    args = parser.parse_args()

    # create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # grab the list of images in the image directory make sure the extension is either .jpg or .png
    image_paths = [os.path.join(args.image_dir, image_path) for image_path in os.listdir(args.image_dir) if image_path.endswith((".jpg", ".png"))]

    # traverse the images in the image directory, make sure the extension if either .jpg or .png
    for image_path in tqdm(image_paths):
        # apply the grab_boundary function to the image
        img_thresholded = laplace_boundary(image_path, prop_black= args.prop_black, bins=args.bin, verbose=args.verbose)

        image_base_path = os.path.basename(image_path)

        if args.present_original:

            # open the image using PIL
            img = PIL.Image.open(image_path)

            # convert the thresholded image to a PIL image
            img_thresholded = PIL.Image.fromarray(img_thresholded)

            # use the draw function to put the thresholded image next to the original image and label them
            img_concat = PIL.Image.new('RGB', (img.width + img_thresholded.width, img.height))
            img_concat.paste(img, (0, 0))
            img_concat.paste(img_thresholded, (img.width, 0))

            # label the images
            img_concat_draw = ImageDraw.Draw(img_concat)
            img_concat_draw.text((0, 0), "Original", fill=(255, 255, 255))
            img_concat_draw.text((img.width, 0), "Laplacian", fill=(255, 255, 255))

            # save the image
            img_concat.save(os.path.join(args.output_dir, image_base_path))

        else:
            raise NotImplementedError("This part of the code is not implemented yet")