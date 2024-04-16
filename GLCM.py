from skimage.feature import graycomatrix, graycoprops
import numpy as np
import cv2


segmented_image = cv2.imread('cleaned2.jpg')
gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

glcm = graycomatrix(gray_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

contrast = graycoprops(glcm, 'contrast')
homogenity = graycoprops(glcm, 'homogeneity')
energy = graycoprops(glcm, 'energy')
correlation = graycoprops(glcm, 'correlation')

print('Contrast: ', contrast)
print('Homogenity: ', homogenity)
print('Energy: ', energy)
print('Correlation: ', correlation)
