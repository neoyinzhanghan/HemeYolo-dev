# save a black_2048_2048.jpg image in the current directory consisting of 2048x2048 black pixels

import numpy as np
import cv2

img = np.zeros((2048, 2048, 3), np.uint8)
cv2.imwrite('black_2048_2048.jpg', img)