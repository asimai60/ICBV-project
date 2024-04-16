import numpy as np
import cv2

def row_by_row_blacking_like_ray_tracking(img):      
    blackened_img = 255 * np.ones_like(img)
    for x in range(2, img.shape[0] - 2):
        black_pxls = np.where(img[x, :] == 0)
        if black_pxls[0].size == 0: continue
        ystart = black_pxls[0][0]
        yend = black_pxls[0][-1]
        blackened_img[x,ystart:yend+1] = 0
    cv2.imshow('row_by_row', blackened_img)
    cv2.waitKey(0)
    return blackened_img

def load_and_resize(path):
    image = cv2.imread(path)
    factor = int(np.round(max(image.shape[0], image.shape[1]) / 1000))
    desired_shape = (image.shape[1]//factor, image.shape[0]//factor)
    image = cv2.resize(image, desired_shape)
    return image

def generate_erosion_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    binary_image = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(binary_image, kernel, iterations=1)
    return erosion

path = 'bottles/17.jpeg'
image = load_and_resize(path)
erosed_image = generate_erosion_image(image)

row_by_row_mask = cv2.bitwise_not(row_by_row_blacking_like_ray_tracking(erosed_image))
segmented_row_by_row = cv2.bitwise_and(image,image,mask=row_by_row_mask)
cv2.imshow('Original Image', image)
cv2.imshow('segmented', segmented_row_by_row)
cv2.waitKey(0)

