import numpy as np
import cv2

image = cv2.imread('bottles/9.jpeg').astype(np.uint8)
factor = int(np.round(max(image.shape[0], image.shape[1]) / 1000))
desired_shape = (image.shape[1]//factor, image.shape[0]//factor)
image = cv2.resize(image, desired_shape)


lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# Split into channels
l, a, b = cv2.split(lab_image)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
enhanced_lab = cv2.merge((cl, a, b))
image = cv2.cvtColor(enhanced_lab, cv2.COLOR_Lab2BGR)



cv2.imshow('Original Image', image)

mask = np.zeros(image.shape[:2], np.uint8)

rect = (1, 1, image.shape[1]- 1, image.shape[0] - 1)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) |(mask == 0), 0,1).astype('uint8')

segmented_image = image * mask2[:,:,np.newaxis]

cv2.imshow('segmented image', segmented_image)
#cv2.imwrite('segmented_image.jpg', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()