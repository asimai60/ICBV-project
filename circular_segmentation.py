import numpy as np
import cv2
import os

def load_image(path, max_size=800):
    image = cv2.imread(path)
    factor = int(np.round(max(image.shape[0], image.shape[1]) / max_size))
    desired_shape = (image.shape[1]//factor, image.shape[0]//factor)
    image = cv2.resize(image, desired_shape)
    return image

def detect_circles(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    edges = cv2.Canny(denoised_image, 20, 0)

    dp = 1
    min_dist = 200
    param1 = 70
    param2 = 70
    min_radius = image.shape[0] // 8
    max_radius = 0

    hough_circles = cv2.HoughCircles(denoised_image, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if hough_circles is None:
        param1 = 50
        param2 = 50
        hough_circles = cv2.HoughCircles(denoised_image, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    return hough_circles

def segment_circles(image, hough_circles):
    mask = 255 * np.ones_like(image)
    x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0]
    if hough_circles is not None:
        mask = np.zeros_like(image)
        hough_circles = np.uint16(np.around(hough_circles))
        for i in hough_circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            x, y = center
            x1, y1 = x - radius, y - radius
            x2, y2 = x + radius, y + radius
            cv2.circle(mask, center, radius, (255, 255, 255), -1)
    
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_and(image, image, mask=mask)
    image = image[y1:y2, x1:x2]
    return image

PATH = 'bottom bottles/'
for im in os.listdir(PATH):
    image = load_image(f'{PATH}{im}')
    hough_circles = detect_circles(image)
    segmented_image = segment_circles(image, hough_circles)
    cv2.imshow(f'Detected Circles in {im}', segmented_image)
    cv2.imwrite(f'segmented bottoms/{im}', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
