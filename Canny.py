import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter
import matplotlib.patches as patches
from skimage.draw import circle_perimeter

plt.close('all')

Local_max_Th = 0.8
LOW_threshold = 35
HIGH_threshold = 70


# Set the directory path
plastic_path = r'C:\Users\nrhot\Downloads\WhatsApp Unknown 2024-04-24 at 12.53.26\PLASTIC'
glass_path = r'C:\Users\nrhot\Downloads\WhatsApp Unknown 2024-04-24 at 12.53.26\GLASS'


# List all files in the directory
files = os.listdir(glass_path)

def load_and_resize(path):
    image = cv2.imread(path)
    factor = int(np.round(max(image.shape[0], image.shape[1]) / 1000))
    desired_shape = (image.shape[1]//factor, image.shape[0]//factor)
    image = cv2.resize(image, desired_shape)
    return image


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blurring to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, LOW_threshold, HIGH_threshold)
    # Display the original image and the edge image
    plt.figure(figsize=(10, 6))

    plt.subplot(121)  # 1 row, 2 columns, 1st subplot = original image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)  # 1 row, 2 columns, 2nd subplot = edge image
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Image')
    plt.axis('off')

    plt.show()
    return edges


def plotCircles(image, detected_circles, bin_size):
    '''
    This function plots the detected circles.
    It draws the circles on top of the original grayscale image.
    `detected_circles` should be an iterable of tuples (radius, y-coordinate, x-coordinate).
    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    ax.title.set_text("Detected Circles")

    for rad, cy, cx in detected_circles:
        circle_patch = patches.Circle((cx * bin_size, cy * bin_size), radius=rad * bin_size, edgecolor=(0, 1, 0), facecolor='none', linewidth=2)
        ax.add_patch(circle_patch)

    plt.show()


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
    cx_buckets = width
    cy_buckets = height
    max_radius = int(np.sqrt(np.square(height) + np.square(width)) / 2)
    rad_buckets = max_radius + 1
    bin_size = 3

    """kernels = generate_circle_kernels(max_radius)
    accumulator = np.zeros((max_radius, *edge_map.shape))
    print("starting HoughCircles")

# Use precomputed kernels for convolution
    for i, kernel in enumerate(kernels):
        accumulator[i] = fftconvolve(edge_map, kernel, mode='same')"""
    accumulator = generate_accumulator(edge_map, max_radius, bin_size)
    print("accumolator")

    local_maxima = find_global_max(accumulator)
    local_maxima = [local_maxima]
    print("Local maxima")

    print(type(local_maxima))
    plotCircles(image, local_maxima, bin_size)
    return local_maxima


# Process each image
for image_file in files:
    # Construct the full path to the image
    image_path = os.path.join(glass_path, image_file)
    print(image_path)
    # Read the image
    image = load_and_resize(image_path)
    # Convert the image to grayscale
    edges = canny(image)
    print("edges")

    local_maxima = HoughCircles(edges, image)

