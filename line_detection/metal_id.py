import numpy as np
from matplotlib import pyplot as plt
import cv2


#PATH = 'Straight_On.jpeg'    

# Convert image to grayscale, formats image to 8-bit
img = cv2.imread('Straight_On.jpeg', cv2.IMREAD_GRAYSCALE)

# implement edge detection with Canny, applied through steps:
# 1. Noise Reduction
# 2. Gradient Calculation
# 3. Non-maximum Suppression
# 4. Double Threshold
# 5. Edge Tracking with Hysteresis
# argument 1 imports the image
# arguement 2 and 3 define the threshold values for obtaining the
# edges that connect the images
ed = cv2.Canny(img, 50, 200, None, 3)

# convert to bgr, prepare image to output in red
ced = cv2.cvtColor(ed, cv2.COLOR_GRAY2BGR)

# obtian lines through Hough transform, vector is outputted
lines = cv.HoughLines(ed, 1, np.pi / 180, 150, None, 0, 0)

# implement Hough Transform
# define vector of 2 floating points

# applies saturation to the image
#sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,1]


#plt.imshow(img, cmap='gray')
#circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=130, param2=30, minRadius=0, maxRadius=0)
#if circles is not None:
#    for x, y, r in circles[0]:
#        c = plt.Circle((x, y), r, fill=False, lw=3, ec='C1')
#        plt.gca().add_patch(c)
#plt.gcf().set_size_inches((12, 8))
cv2.imwrite("Straight_On_ed.png",ed)
cv2.imwrite("Straight_On_ced.png",ced)
