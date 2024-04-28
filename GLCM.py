from skimage.feature import graycomatrix, graycoprops
import numpy as np
import cv2
import os

paths = os.listdir('segmented bottoms')

# def sort_key(path):
#     if path.split('.')[0][-1] == 'p':
#         return int(path.split('.')[0][:-1])
#     else:
#         return int(path.split('.')[0][:-1]) * 100
# paths.sort(key=sort_key)

for img_name in paths:
    image = cv2.imread(f'segmented bottoms/{img_name}')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'correlation')
    print('contrast of '+img_name,np.mean(contrast.flatten()))

