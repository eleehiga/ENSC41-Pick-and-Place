import numpy as np
import math
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
# create copy for probabilittic hough transform
cedp = np.copy(ced)

# obtian lines through Hough transform, vector is outputted
lines = cv2.HoughLines(ed, 1, np.pi / 180, 150, None, 0, 0)

# display lines from transform
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(ced, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

# apply probabilitic hough transform
linesP = cv2.HoughLinesP(ed, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
         l = linesP[i][0]
         cv2.line(cedp, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)


# applies saturation to the image
#sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,1]


#plt.imshow(img, cmap='gray')
#circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=130, param2=30, minRadius=0, maxRadius=0)
#if circles is not None:
#    for x, y, r in circles[0]:
#        c = plt.Circle((x, y), r, fill=False, lw=3, ec='C1')
#        plt.gca().add_patch(c)
#plt.gcf().set_size_inches((12, 8))
cv2.imwrite("Straight_On_ced.png",ced)
cv2.imwrite("Straight_On_cedp.png",cedp)