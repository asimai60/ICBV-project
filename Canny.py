import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import feature
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter
from skimage.draw import circle_perimeter

plt.close('all')
# Set the directory path
plastic_path = r'C:\Users\nrhot\Downloads\WhatsApp Unknown 2024-04-24 at 12.53.26\PLASTIC'
glass_path = r'C:\Users\nrhot\Downloads\WhatsApp Unknown 2024-04-24 at 12.53.26\GLASS'


Local_max_Th = 1
LOW_threshold = 45
HIGH_threshold = 90
bin_size = 3
radius = 15






def load_and_resize(path):
    image = cv2.imread(path)
    factor = int(np.round(max(image.shape[0], image.shape[1]) / 1000))
    desired_shape = (image.shape[1]//factor, image.shape[0]//factor)
    image = cv2.resize(image, desired_shape)
    return image


def canny(image):

    # Apply Gaussian Blurring to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, LOW_threshold, HIGH_threshold)
    # Display the original image and the edge image

    return edges


def plotCircles(image, detected_circles, bin_size):
    '''
    This function plots the detected circles.
    It draws the circles on top of the original grayscale image.
    `detected_circles` should be an iterable of tuples (radius, y-coordinate, x-coordinate).
    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    ax.title.set_text(image_path[70:])
    mask = np.zeros_like(image)
    for rad, cy, cx in detected_circles:
        cv2.circle(mask, (int(cx * bin_size), int(cy * bin_size)), int(rad * bin_size) + radius, (255, 255, 255), -1)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_and(image, image, mask=mask)
    plt.imshow(image)
    plt.show()
    return image


def generate_circle_kernels(max_radius):
    # Create a grid of coordinates
    grid_size = max_radius * 2
    center = max_radius
    y, x = np.ogrid[-center:grid_size-center, -center:grid_size-center]
    squared_distances = x**2 + y**2

    # Create a list to hold all kernels
    kernels = []

    # Generate a mask for each circle radius
    for radius in range(1, max_radius + 1):
        mask = np.logical_and(squared_distances >= (radius - 0.5)**2, squared_distances <= (radius + 0.5)**2)
        kernel = np.zeros((grid_size, grid_size))
        kernel[mask] = 1
        kernels.append(kernel)

    return kernels


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


def find_global_max(accumulator):
    # Flatten the accumulator array
    flat_accumulator = accumulator.flatten()

    # Find the index of the maximum value
    max_index = np.argmax(flat_accumulator)

    # Convert the flattened index to multi-dimensional index
    max_index_multi = np.unravel_index(max_index, accumulator.shape)

    return max_index_multi



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


def HoughCircles(edge_map, image):
    # Set the radius, cx and cy range
    height, width = edge_map.shape
    max_radius = width // 2

    accumulator = generate_accumulator(edge_map, max_radius, bin_size)

    local_maxima = find_global_max(accumulator)
    local_maxima = [local_maxima]

    masked_image = plotCircles(image, local_maxima, bin_size)
    return masked_image


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


data = []
path = plastic_path
# List all files in the directory
files = os.listdir(path)
print("plastic data:")
# Process each image
for image_file in files:
    # Construct the full path to the image
    image_path = os.path.join(path, image_file)
    # Read the image
    image = load_and_resize(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = canny(gray)
    masked_image = HoughCircles(edges, image)
    gray_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    data.append(feature_extraction(gray_masked))

print("glass data")
path = glass_path
# List all files in the directory
files = os.listdir(path)
for image_file in files:
    # Construct the full path to the image
    image_path = os.path.join(path, image_file)
    # Read the image
    image = load_and_resize(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = canny(gray)
    masked_image = HoughCircles(edges, image)
    gray_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    data.append(feature_extraction(gray_masked))

pca_features = data

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pca_features = np.array(pca_features)

# Third principal component

labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # Binary labels for simplicity
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
z = pca_features[:, 4]

# Plotting data, colored by label
scatter = ax.scatter(x, y, z, c=colors, marker='o')

# Create a legend (optional)
red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Label 0')
blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Label 1')
plt.legend(handles=[red_patch, blue_patch])

# Title and show
plt.title('3D PCA Results')
plt.show()