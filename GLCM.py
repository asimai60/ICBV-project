from skimage.feature import graycomatrix, graycoprops
import numpy as np
import cv2

path_plastic = input('Enter the path of the plastic image: ')
path_glass = input('Enter the path of the glass image: ')

segmented_plastic = cv2.imread(f'segmented bottoms/{path_plastic}.jpeg')
segmented_glass = cv2.imread(f'segmented bottoms/{path_glass}.jpeg')
gray_image_plastic = cv2.cvtColor(segmented_plastic, cv2.COLOR_BGR2GRAY)
gray_image_glass = cv2.cvtColor(segmented_glass, cv2.COLOR_BGR2GRAY)

glcm_plastic = graycomatrix(gray_image_plastic, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
glcm_glass = graycomatrix(gray_image_glass, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

contrast_plastic = graycoprops(glcm_plastic, 'contrast')
homogenity_plastic = graycoprops(glcm_plastic, 'homogeneity')
energy_plastic = graycoprops(glcm_plastic, 'energy')
correlation_plastic = graycoprops(glcm_plastic, 'correlation')


contrast_glass = graycoprops(glcm_glass, 'contrast')
homogenity_glass = graycoprops(glcm_glass, 'homogeneity')
energy_glass = graycoprops(glcm_glass, 'energy')
correlation_glass = graycoprops(glcm_glass, 'correlation')


print(f'Contrast plastic {path_plastic}: ', contrast_plastic)
print(f'Contrast glass {path_glass}: ', contrast_glass)

print(f'Homogenity plastic {path_plastic}: ', homogenity_plastic)
print(f'Homogenity glass {path_glass}: ', homogenity_glass)

print(f'Energy plastic {path_plastic}: ', energy_plastic)
print(f'Energy glass {path_glass}: ', energy_glass)

print(f'Correlation plastic {path_plastic}: ', correlation_plastic)
print(f'Correlation glass {path_glass}: ', correlation_glass)

