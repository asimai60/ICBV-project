import os
import numpy as np
import cv2

def get_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(img, contours):
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    return img
 
def who_is_black2(img):
    for i in range(66):
        blackend_img = 255*np.ones_like(img)
        for x in range(2, img.shape[0] - 2):
            for y in range(2, img.shape[1] - 2):
                whites_Around = cv2.countNonZero(img[x - 2:x + 2, y - 2:y + 2])
                if whites_Around/25 < 0.35:
                    blackend_img[x, y] = 0
                else:
                    blackend_img[x, y] = img[x, y]
        img = blackend_img
        # cv2.imshow('Blacked', cv2.resize(blackend_img,desired_shape))
        # cv2.waitKey(0)
    return img

def who_is_black(img, k=5):
    kernel = np.ones((k, k), np.float32)
    
    white_neighbors = cv2.filter2D((img > 0).astype(np.float32), -1, kernel)

    total_pixels = k * k

    black_threshold = 0.35 * total_pixels
    white_threshold = 0.45 * total_pixels

    img[(white_neighbors < black_threshold) & (img > 0)] = 0
    img[(img == 0) & (white_neighbors > white_threshold)] = 255

    return img

def remove_artifacts(img, k=5):
    kernel = np.ones((k, k), np.float32)

    for i in range (5):
        neighbors = cv2.filter2D((img > 0).astype(np.float32), -1, kernel)

        total_pixels = k * k

        threshold = 0.3 * total_pixels

        img[(neighbors > threshold) & (img == 0)] = 255

    return img

def keep_largest_blob(binary_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8, ltype=cv2.CV_32S)
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    output_image = np.zeros_like(binary_mask)
    output_image[labels == largest_component] = 255

    return output_image

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

image = cv2.imread('bottles/17.jpeg')

factor = int(np.round(max(image.shape[0], image.shape[1]) / 1000))
desired_shape = (image.shape[1]//factor, image.shape[0]//factor)
image = cv2.resize(image, desired_shape)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
denoised_image = cv2.GaussianBlur(gray_image, (5,5), 0)

binary_image = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(binary_image, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel, iterations=1)
blurred = cv2.GaussianBlur(denoised_image, (7, 7), 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast_enhanced = clahe.apply(denoised_image)
edges = cv2.Canny(binary_image, 100, 200)


# cv2.imshow('Original Image', cv2.resize(image,desired_shape))
# cv2.imshow('Gray Image', cv2.resize(gray_image,desired_shape))
# cv2.imshow('Denoised Image', cv2.resize(denoised_image,desired_shape))
# cv2.imshow('Edges', cv2.resize(edges,desired_shape))
# cv2.imshow('Binary Image', cv2.resize(binary_image,desired_shape))
cv2.imshow('Erosion', cv2.resize(erosion,desired_shape))
# cv2.imshow('Dilation', cv2.resize(dilation,desired_shape))
# cv2.imshow('Blurred', cv2.resize(blurred,desired_shape))
# cv2.imshow('Contrast Enhanced', cv2.resize(contrast_enhanced,desired_shape))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.waitKey(0)

row_by_row_mask = cv2.bitwise_not(row_by_row_blacking_like_ray_tracking(erosion))

# blacked2 = who_is_black2(erosion)
# cv2.imshow('Blacked2', cv2.resize(blacked2,desired_shape))
# cv2.waitKey(0)
# mask = cv2.bitwise_not(blacked2)
# cv2.imshow('Mask', cv2.resize(mask,desired_shape))
# cv2.waitKey(0)
# largest_blob = keep_largest_blob(mask)

# segmented = cv2.bitwise_and(image, image, mask=mask)
# fully_segmented = cv2.bitwise_and(image, image, mask=largest_blob)

# diff = cv2.bitwise_xor(fully_segmented, segmented)

segmented_row_by_row = cv2.bitwise_and(image,image,mask=row_by_row_mask)


cv2.imshow('Original Image', cv2.resize(image,desired_shape))
# cv2.imshow('Largest Blob', cv2.resize(largest_blob,desired_shape))
# cv2.imshow('Fully Segmented', cv2.resize(fully_segmented,desired_shape))
# cv2.imshow('Segmented', cv2.resize(segmented,desired_shape))
# cv2.imshow('Difference', cv2.resize(diff,desired_shape))
cv2.imshow('segmented', cv2.resize(segmented_row_by_row,desired_shape))

cv2.waitKey(0)

