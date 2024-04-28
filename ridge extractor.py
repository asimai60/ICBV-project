import os
import cv2
import numpy as np

PATH = 'segmented bottoms/'

for im in os.listdir(PATH):
    image = cv2.imread(f'{PATH}{im}')
    original = image.copy()
    inner_circle = cv2.circle(image, (image.shape[1]//2, image.shape[0]//2), image.shape[0]//2 - image.shape[0]//8, (0, 0, 0),-1)
    gray_image = cv2.cvtColor(inner_circle, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.GaussianBlur(gray_image, (9,9), 0)
    LoG = cv2.Laplacian(denoised_image, cv2.CV_64F, ksize=5)
    cv2.imshow('LoG', LoG)
    cv2.imshow('Original', original)
    cv2.imwrite(f'ridge patterns/{im.split(".")[0]}_ridge.jpeg', LoG)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv2.destroyAllWindows()

