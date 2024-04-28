import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import feature


plastic_path = r"C:\Users\nrhot\Downloads\WhatsApp Unknown 2024-04-24 at 12.53.26\cropped\plastic"
glass_path = r'C:\Users\nrhot\Downloads\WhatsApp Unknown 2024-04-24 at 12.53.26\cropped\glass'
LOW_threshold = 20
HIGH_threshold = 70
threshold = 30
RHO = 1
THETA = np.pi/ 45
LINESTH = 70


def plot_circles(circles, image):
    # Convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    output =  np.copy(image)# Make a copy of the image

    # Loop over the circles
    for (x, y, r) in circles:
        # Draw the circle in the output image
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        # Draw a rectangle (center point) in the output image
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # Display the result in a matplotlib window
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title('Detected Circles')
    plt.axis('off')
    plt.show()


def load_and_resize(path):
    image = cv2.imread(path)
    factor = int(np.round(max(image.shape[0], image.shape[1]) / 1000))
    desired_shape = (480, 480)
    image = cv2.resize(image, desired_shape)
    return image


def feature_extraction(image):
    # Convert gray image to uint8
    gray_image = (image * 255).astype(np.uint8)
    glcm = feature.graycomatrix(gray_image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, symmetric=True,
                                normed=True)

    # Extract properties from GLCM
    contrast = feature.graycoprops(glcm, 'contrast')
    dissimilarity = feature.graycoprops(glcm, 'dissimilarity')
    homogeneity = feature.graycoprops(glcm, 'homogeneity')
    energy = feature.graycoprops(glcm, 'energy')
    correlation = feature.graycoprops(glcm, 'correlation')
    print(f'Contrast: {contrast}')
    print(f'Dissimilarity: {dissimilarity}')
    print(f'Homogeneity: {homogeneity}')
    print(f'Energy: {energy}')
    print(f'Correlation: {correlation}')
    data = [contrast, dissimilarity, homogeneity, energy, correlation]
    return data



def canny(gray):
    # Apply Gaussian Blurring to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    #blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, LOW_threshold, HIGH_threshold)
    # Display the original image and the edge image
    return edges


def polar_to_cartesian(rho, theta):
    """ Convert polar coordinates to Cartesian 'Ax + By = C' format. """
    A = np.cos(theta)
    B = np.sin(theta)
    C = rho
    return A, B, C


def intersection_point(line1, line2):
    """ Find intersection point of two lines given in 'Ax + By = C' format. """
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    determinant = A1*B2 - A2*B1
    if determinant == 0:
        # Lines are parallel
        return None
    else:
        x = (B2*C1 - B1*C2) / determinant
        y = (A1*C2 - A2*C1) / determinant
        return (x, y)


def find_intersections(lines):
    """ Find and categorize intersections of multiple lines. """
    intersections = {}
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = intersection_point(lines[i], lines[j])
            if pt:
                # Round the point coordinates to avoid floating point precision issues
                rounded_point = (round(pt[0], 2), round(pt[1], 2))
                if rounded_point in intersections:
                    intersections[rounded_point] += 1
                else:
                    intersections[rounded_point] = 1
    return intersections


def detect_lines(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]

    # Calculate the cropping coordinates
    crop_size = 1/8  # Fraction to crop from each side
    x_start = int(w * crop_size)
    x_end = int(w * (1 - crop_size))
    y_start = int(h * crop_size)
    y_end = int(h * (1 - crop_size))

    # Crop the image 
    gray_cropped = gray[y_start:y_end, x_start:x_end]
    # Apply edge detection using the Canny edge detector
    edges = canny(gray_cropped)
    edges_full = canny(gray)

    # Use HoughLines to detect lines in the edge map
    lines = cv2.HoughLines(edges, RHO, THETA, LINESTH)  # These parameters may need adjustment for your specific case

    # Create a copy of the original image to draw lines on
    result = image.copy()

    if lines is not None:
        cartesian_lines = [polar_to_cartesian(rho, theta) for rho, theta in lines[:,0]]
        intersections = find_intersections(cartesian_lines)
        if intersections:
            max_point = max(intersections, key=intersections.get)
            max_value = intersections[max_point]
            intersections.pop(max_point)
            if intersections:
                sec_max_point = max(intersections, key=intersections.get)
                distance = np.sqrt((max_point[0] - sec_max_point[0])**2 + (max_point[1] - sec_max_point[1])**2)
                if distance < threshold:
                    max_value = max_value + intersections[sec_max_point]

            print("Maximum value is at point:", max_point)
            print("Number of lines intersecting:", max_value)
        else:
            print("no interesting lines")
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Show the result
        plt.figure(figsize=(10, 6))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Detected Lines')
        plt.axis('off')

        plt.show()
    else:
        print("no lines detected")
        # Show the result

        # hough_circles = cv2.HoughCircles(edges_full, cv2.HOUGH_GRADIENT, 1, 80, param1=50, param2=50, minRadius=100, maxRadius=0)
        # if hough_circles is not None:
        #     plot_circles(hough_circles, image)
        # else:
        #     print('No circles detected!')
        plt.figure(figsize=(10, 6))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
        # plt.imshow(cv2.cvtColor(edges_full, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Detected Lines')
        plt.axis('off')

        plt.show()




def main(path):
    files = os.listdir(path)
    # Process each image
    for image_file in files:
        # Construct the full path to the image
        image_path = os.path.join(path, image_file)
        # Read the image
        image = load_and_resize(image_path)
        gray_masked = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_masked = canny(gray_masked)
        shape = gray_masked.shape[0]
        lines = detect_lines(image)
        """circles = cv2.HoughCircles(gray_masked, cv2.HOUGH_GRADIENT, 1, 15, param1=50, param2=50, minRadius=1, maxRadius=shape//8)
        # If at least one circle is detected
        if circles is not None:
            plot_circles(circles, image)
        else:
            print('No circles detected!')"""




print("glass data")
main(glass_path)
print("plastic data:")
main(plastic_path)



"""pca_features = data

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pca_features = np.array(pca_features)

# Third principal component

labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]*4)  # Binary labels for simplicity
print(len(labels) == 26)
print("plastic avarage:")
plastic_ava = np.mean(pca_features[:16], axis=0)
print(plastic_ava)
print("glass average:")
glass_ava = np.mean(pca_features[16:], axis=0)
print(glass_ava)

# Create a color map based on labels
colors = ['red' if label == 0 else 'blue' for label in labels]  # Red for label 0, blue for label 1
pca_1 = pca_features[:,:1,:]
# Create a new figure for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pca_1 = np.squeeze(pca_1)
print(pca_1.shape)
# Extract individual components
x = pca_features[:, 1]
y = pca_features[:, 2]
#z = pca_features[:, 4]

# Plotting data, colored by label
scatter = ax.scatter(x, y, c=colors, marker='o')

# Create a legend (optional)
red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Label 0')
blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Label 1')
plt.legend(handles=[red_patch, blue_patch])

# Title and show
plt.title('3D PCA Results')
plt.show()"""
