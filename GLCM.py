from skimage.feature import graycomatrix, graycoprops
import numpy as np
import cv2


segmented_13 = cv2.imread('segmented bottoms/10.jpeg')
segmented_19 = cv2.imread('segmented bottoms/12.jpeg')
gray_image13 = cv2.cvtColor(segmented_13, cv2.COLOR_BGR2GRAY)
gray_image19 = cv2.cvtColor(segmented_19, cv2.COLOR_BGR2GRAY)

glcm13 = graycomatrix(gray_image13, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
glcm19 = graycomatrix(gray_image19, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

contrast13 = graycoprops(glcm13, 'contrast')
homogenity13 = graycoprops(glcm13, 'homogeneity')
energy13 = graycoprops(glcm13, 'energy')
correlation13 = graycoprops(glcm13, 'correlation')


contrast19 = graycoprops(glcm19, 'contrast')
homogenity19 = graycoprops(glcm19, 'homogeneity')
energy19 = graycoprops(glcm19, 'energy')
correlation19 = graycoprops(glcm19, 'correlation')


print('Contrast13: ', contrast13)
print('Contrast19: ', contrast19)

print('Homogenity13: ', homogenity13)
print('Homogenity19: ', homogenity19)

print('Energy13: ', energy13)
print('Energy19: ', energy19)

print('Correlation13: ', correlation13)
print('Correlation19: ', correlation19)
