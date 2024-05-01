import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def get_pixel(img, center, x, y):
    try:
        return 1 if img[x][y] >= center else 0
    except IndexError:  # Use specific exception handling
        return 0

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    values = [get_pixel(img, center, x + dx, y + dy) 
              for dx, dy in [(-1, -1), (-1, 0), (-1, 1), 
                             (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]]
    # Convert binary values to decimal using bit manipulation
    return sum(val * (1 << i) for i, val in enumerate(values))


PATH = 'segmented bottoms/'
for im in os.listdir(PATH):
    image = cv2.imread(PATH+im)
    img_bgr = cv2.circle(image, (image.shape[1]//2, image.shape[0]//2), image.shape[0]//2 - image.shape[0]//8, (0, 0, 0),-1)


    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce image noise if it's required
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    cv2.imshow('Blurred', blurred)
    # Use a threshold to create a binary image
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow('Threshold', thresh)
    # Apply a morphological operation to close the gaps in the image
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imshow('Closed', closed)
    # Find contours in the image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw the contours on the original image
    cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), 3)

    cv2.imshow('Contours', img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape

    # Initialize the LBP image
    img_lbp = np.zeros((height, width), np.uint8)

    for i in range(height):
        for j in range(width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

    threshold = 255 // 2
    img_lbp = cv2.threshold(img_lbp, threshold, 255, cv2.THRESH_BINARY)[1]

    blur = cv2.GaussianBlur(img_lbp, (9, 9), 0)
    cv2.imshow('Blurred', blur)

    img_lbp = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    hough_circles = cv2.HoughCircles(img_lbp, cv2.HOUGH_GRADIENT, 1, 10, param1=125, param2=125, minRadius=img_lbp.shape[0]//8, maxRadius=0)
    if hough_circles is not None:
        hough_circles = np.uint16(np.around(hough_circles)) 
        for i in hough_circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img_bgr, center, radius, (0, 255, 0), 3)


    # Display the original and LBP images
    cv2.imshow('Original', img_bgr)
    cv2.imshow('LBP', img_lbp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



