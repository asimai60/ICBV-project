import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import time
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter
from skimage.draw import circle_perimeter

plastic_path = 'Tom bottles/plastic cropped'
glass_path = 'Tom bottles/glass cropped'
LOW_threshold = 20
HIGH_threshold = 70
threshold = 30
RHO = 1
THETA = np.pi/ 45
LINESTH = 75

Local_max_Th = 0.6
LOW2_threshold = 30
HIGH2_threshold = 150
bin_size = 3
K_size = (5, 5)


def plot_circles(circles, image):
    # Convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    output = np.copy(image)  # Make a copy of the image

    # Loop over the circles
    for (x, y, r) in circles:
        # Draw the circle in the output image
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        # Draw a rectangle (center point) in the output image
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


def load_and_resize(path):
    image = cv2.imread(path)
    desired_shape = (480, 480)
    image = cv2.resize(image, desired_shape)
    return image


def canny(gray):
    # Apply Gaussian Blurring to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, LOW_threshold, HIGH_threshold)
    # Display the original image and the edge image
    return edges


def polar_to_cartesian(rho, theta):
    """ Convert polar coordinates to Cartesian 'Ax + By = C' format. """
    a = np.cos(theta)
    b = np.sin(theta)
    c = rho
    return a, b, c


def intersection_point(line1, line2):
    """ Find intersection point of two lines given in 'Ax + By = C' format. """
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    determinant = A1 * B2 - A2 * B1
    if determinant == 0:
        # Lines are parallel
        return None
    else:
        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        return x, y


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


    # Use HoughLines to detect lines in the edge map
    lines = cv2.HoughLines(edges, RHO, THETA, LINESTH)  # These parameters may need adjustment for your specific case

    # Create a copy of the original image to draw lines on
    result = image.copy()

    if lines is not None:
        """cartesian_lines = [polar_to_cartesian(rho, theta) for rho, theta in lines[:,0]]
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
            print("no interesting lines")"""
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
        return True
    else:
        return False


def canny2(gray):
    # Apply Gaussian Blurring to reduce noise and improve edge detection
    center = gray.shape[0]//2
    inner_circle = cv2.circle(gray, (center, center), center - gray.shape[0]//8, (0, 0, 0),-1)
    blurred = cv2.GaussianBlur(inner_circle, K_size, 0)
    # Use adaptive thresholding to create a binary image
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(adaptive_thresh, LOW2_threshold, HIGH2_threshold)
    return edges


def HoughCircles(image, path):
    # Set the radius, cx and cy range
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_map = canny2(gray)
    height, width = edge_map.shape
    max_radius = width // 2

    accumulator = generate_accumulator(edge_map, max_radius, bin_size)

    local_maxima = find_local_maxima(accumulator, Local_max_Th)
    # Define the value you want to remove
    value_to_remove = np.array([79, 81, 81])

    # Create a condition where the second and third columns are not both 81
    mask = ~(local_maxima[:, 1] == 81) | ~(local_maxima[:, 2] == 81)

    # Apply the mask to 'local_maxima' to filter out the unwanted row
    local_maxima = local_maxima[mask]

    return local_maxima


def generate_accumulator(edge_map, max_radius, bin_size=1):
    # Use uniform_filter to effectively "bin" the image
    # by computing the local mean over the areas of size (bin_size x bin_size)
    # This is much more efficient than manually summing up the pixels
    if bin_size > 1:
        edge_map = uniform_filter(edge_map, size=bin_size)
        edge_map = edge_map[::bin_size, ::bin_size]

    # Initialize the accumulator array for the binned edge map
    accumulator = np.zeros((max_radius, *edge_map.shape))

    # Create a two-dimensional grid of coordinates for the binned edge map
    y, x = np.indices(edge_map.shape)
    max_radius_binned = max_radius // bin_size + 1
    # Generate circle masks and accumulate
    for radius in range(1, max_radius_binned):
        # Generate the mask for this radius
        mask = np.zeros_like(edge_map, dtype=float)
        rr, cc = circle_perimeter(edge_map.shape[0]//2, edge_map.shape[1]//2, radius)

        # Ensure that the indices are within the dimensions of 'mask'
        rr = np.clip(rr, 0, mask.shape[0] - 1)
        cc = np.clip(cc, 0, mask.shape[1] - 1)

        mask[rr, cc] = 1

        # Convolve and accumulate the results in the corresponding layer of the accumulator
        accumulator[radius - 1] = fftconvolve(edge_map, mask, mode='same')

    return accumulator


def find_local_maxima(accumulator, threshold=0.5, neighborhood_size=3):
    threshold_abs = threshold * np.max(accumulator)
    local_maxima = np.zeros_like(accumulator, dtype=bool)

    # Iterate over each pixel in the accumulator
    for index, value in np.ndenumerate(accumulator):
        if value >= threshold_abs:
            # Define the neighborhood boundaries
            min_bound = np.maximum(np.subtract(index, (neighborhood_size // 2,)*len(index)), 0)
            max_bound = np.minimum(np.add(index, (neighborhood_size // 2 + 1,)*len(index)), accumulator.shape)

            neighborhood = accumulator[min_bound[0]:max_bound[0], min_bound[1]:max_bound[1]]

            # Check if the current pixel is the maximum within its neighborhood
            if value == np.max(neighborhood):
                local_maxima[index] = True
    peaks_indices = np.argwhere(local_maxima)
    return peaks_indices


def main(path, label):
    files = os.listdir(path)
    amuont = len(files)
    not_detected = 0
    correct = 0
    # Process each image
    for image_file in files:
        # Construct the full path to the image
        image_path = os.path.join(path, image_file)
        # Read the image
        image = load_and_resize(image_path)
        lines = detect_lines(image)
        if not lines:
            circls = HoughCircles(image,image_path)
            if circls.any():
                print("found_circles")
                if label == "glass":
                    correct = correct + 1
                else:
                    print(image_path[70:])
            else:
                if label == "plastic":
                    correct = correct + 1
                else:
                    print(image_path[70:])

            not_detected += 1

        else:
            if label == "plastic":
                correct = correct + 1
            else:
                print(image_path[70:])
    time.sleep(2)

    return amuont, correct



print("glass data")
amount_g , pglass= main(glass_path, "glass")
print("plastic data:")
amount_p, prate = main(plastic_path, "plastic")
print("prate:", prate)
print("pglass:", pglass)
print("succses rate", (prate + pglass) / (amount_g + amount_p) * 100)





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
