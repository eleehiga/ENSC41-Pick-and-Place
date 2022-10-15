import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,50).__str__()
import cv2
import numpy as np
from PIL import Image

def get_size_format(b, factor=1024, suffix="B"):
    """
    Scale bytes to its proper byte format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"

def compress_img(image_name, new_size_ratio=0.9, quality=90, width=None, height=None, to_jpg=True):
    # load the image to memory
    img = Image.open(image_name)
    # print the original image shape
    print("[*] Image shape:", img.size)
    # get the original image size in bytes
    image_size = os.path.getsize(image_name)
    # print the size before compression/resizing
    print("[*] Size before compression:", get_size_format(image_size))
    if new_size_ratio < 1.0:
        # if resizing ratio is below 1.0, then multiply width & height with this ratio to reduce image size
        img = img.resize((int(img.size[0] * new_size_ratio), int(img.size[1] * new_size_ratio)), Image.ANTIALIAS)
        # print new image shape
        print("[+] New Image shape:", img.size)
    elif width and height:
        # if width and height are set, resize with them instead
        img = img.resize((width, height), Image.ANTIALIAS)
        # print new image shape
        print("[+] New Image shape:", img.size)
    # split the filename and extension
    filename, ext = os.path.splitext(image_name)
    # make new filename appending _compressed to the original file name
    if to_jpg:
        # change the extension to JPEG
        new_filename = f"{filename}_compressed.jpg"
    else:
        # retain the same extension of the original image
        new_filename = f"{filename}_compressed{ext}"
    try:
        # save the image with the corresponding quality and optimize set to True
        img.save(new_filename, quality=quality, optimize=True)
    except OSError:
        # convert the image to RGB mode first
        img = img.convert("RGB")
        # save the image with the corresponding quality and optimize set to True
        img.save(new_filename, quality=quality, optimize=True)
    print("[+] New file saved:", new_filename)
    # get the new image size in bytes
    new_image_size = os.path.getsize(new_filename)
    # print the new size in a good format
    print("[+] Size after compression:", get_size_format(new_image_size))
    # calculate the saving bytes
    saving_diff = new_image_size - image_size
    # print the saving percentage
    print(f"[+] Image size change: {saving_diff/image_size*100:.2f}% of the original image size.")

#''' # the examples on stack overflow had a green background for their metallic object
# read input
def identify_alum(img):
    img = cv2.imread('Straight_On.png', cv2.IMREAD_UNCHANGED)

    # convert to hsv and get saturation channel
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,1]

    # do a little Gaussian filtering
    blur = cv2.GaussianBlur(sat, (3,3), 0)
    #(3,3) is the Gaussian kernel, 
    # blurring the image mitigates noise

    # threshold and invert to create initial mask
    mask = 255 - cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
    # 100 is threshold val, 255 is maxval, THRESH_Bin is thresholding type
    # if >100 then set image's pixel to max, else set to 0

    # apply morphology close to fill interior regions in mask
    kernel = np.ones((15,15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # get outer contours from inverted mask and get the largest (presumably only one due to morphology filtering)
    cntrs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    result = img.copy()
    area_thresh = 0
    for c in cntrs:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area = area_thresh
            big_contour = c

    # draw largest contour
    cv2.drawContours(result, [big_contour], -1, (0,0,255), 2)


    # write result to disk
    cv2.imwrite("shiny_mask.png", mask)
    cv2.imwrite("shiny_outline.png", result)

    # display it
    cv2.imshow("IMAGE", img)
    cv2.imshow("MASK", mask)
    cv2.imshow("RESULT", result)
    cv2.waitKey(0)
#'''
'''
# Load image, convert to grayscale, Gaussian Blur, adaptive threshold
image = cv2.imread('Straight_On.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (13,13), 0)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,7)

# Morph close
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

# Find contours, sort for largest contour, draw contour
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
for c in cnts:
    cv2.drawContours(image, [c], -1, (36,255,12), 2)
    break

cv2.imshow('thresh', thresh)
cv2.imshow('image', image)
cv2.waitKey()
'''


