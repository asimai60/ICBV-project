import cv2
import numpy as np

# Load images
img1 = cv2.imread('ridge patterns/template_ridge.jpeg', cv2.IMREAD_GRAYSCALE)  # Query image
img2 = cv2.imread('ridge patterns/20_ridge.jpeg', cv2.IMREAD_GRAYSCALE)   # Train image

if img1 is None or img2 is None:
    raise ValueError("One of the images didn't load. Check the file path and file integrity.")

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the results
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
