import cv2
import numpy as np
import os

def load_and_convert(image_path):
    """ Load an image and convert it to grayscale. """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the reference pattern image
pattern_img = load_and_convert('ridge patterns/template_ridge.jpeg')

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors in the pattern image
pattern_kp, pattern_des = sift.detectAndCompute(pattern_img, None)

# List of target images
target_images = [f'ridge patterns/{path}' for path in os.listdir('ridge patterns')]
similarity_scores = []

# Matcher using FLANN based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

for target_path in target_images:
    target_img = load_and_convert(target_path)
    target_kp, target_des = sift.detectAndCompute(target_img, None)

    # Matching descriptor using KNN algorithm
    matches = flann.knnMatch(pattern_des, target_des, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Calculate a score (could be simply the number of good matches)
    score = len(good_matches)
    similarity_scores.append((target_path, score))

# Sort images by descending similarity score
similarity_scores.sort(key=lambda x: x[1], reverse=True)

# Display the scores
for img_path, score in similarity_scores:
    print(f"Image: {img_path}, Similarity Score: {score}")
