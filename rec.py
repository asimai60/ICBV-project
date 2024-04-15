import numpy as np
import cv2
image = cv2.imread('depth_map.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

binary_image = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(binary_image, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel, iterations=1)
blurred = cv2.GaussianBlur(denoised_image, (7, 7), 0)
high_pass = cv2.subtract(denoised_image, blurred)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast_enhanced = clahe.apply(denoised_image)

edges = cv2.Canny(gray_image, 100, 200)
#cv2.imshow('Original Image', cv2.resize(image,(540,960)))
#cv2.imshow('Gray Image', cv2.resize(gray_image,(540,960)))
#cv2.imshow('Denoised Image', cv2.resize(denoised_image,(540,960)))
#cv2.imshow('Edges', cv2.resize(edges,(540,960)))
#cv2.imshow('Binary Image', cv2.resize(binary_image,(540,700)))
#cv2.imshow('Erosion', cv2.resize(erosion,(540,700)))
#cv2.waitKey(0)
#cv2.imshow('Dilation', cv2.resize(dilation,(540,700)))
#cv2.imshow('Blurred', cv2.resize(blurred,(540,960)))
#cv2.imshow('High Pass', cv2.resize(high_pass,(540,960)))
#cv2.imshow('Contrast Enhanced', cv2.resize(contrast_enhanced,(540,960)))

def who_is_black(img):
    kernel = np.ones((5, 5))  # Create a 5x5 matrix of ones
    for x in range(2, img.shape[0] - 2):
        for y in range(2, img.shape[1] - 2):
            whites_Around = cv2.countNonZero(img[x - 2:x + 2, y - 2:y + 2])
            if whites_Around/25 < 0.35:
                img[x, y] = 0
    return img
cv2.imshow('Who is black', cv2.resize(who_is_black(erosion),(540,700)))

                
            


cv2.waitKey(0)