import numpy as np
import cv2
import os


PATH = 'bottom bottles/'
for im in os.listdir(PATH):
    print(im)
    image = cv2.imread(f'{PATH}{im}')

    reshape_factor = max(image.shape[0], image.shape[1]) // 1000
    desired_shape = (image.shape[1]//reshape_factor, image.shape[0]//reshape_factor)

    image = cv2.resize(image, desired_shape)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    edges = cv2.Canny(denoised_image, 20, 0)

    dp = 1
    min_dist = 200
    param1 = 70
    param2 = 70
    min_radius = image.shape[0] //8
    max_radius = 0

    hough_circles = cv2.HoughCircles(denoised_image, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

    if hough_circles is not None:
        hough_circles = np.uint16(np.around(hough_circles))
        mask = np.zeros_like(image)
        for i in hough_circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(mask, center, radius, (255, 255, 255), -1)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        image = cv2.bitwise_and(image, image, mask=mask)
        
        cv2.imshow('Detected Circles', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        param1 = 50
        param2 =50
        hough_circles = cv2.HoughCircles(denoised_image, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
        if hough_circles is not None:
            hough_circles = np.uint16(np.around(hough_circles))
            mask = np.zeros_like(image)
            for i in hough_circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                cv2.circle(mask, center, radius, (255, 255, 255), -1)

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            image = cv2.bitwise_and(image, image, mask=mask)
            
            cv2.imshow('Detected Circles', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f'No circles detected in image {im}')