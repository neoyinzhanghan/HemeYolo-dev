import cv2
import numpy as np

# save an black image of 2048x2048 pixels in the current directory


image = np.zeros((2048, 2048, 3), np.uint8)
cv2.imwrite("black.jpg", image)
