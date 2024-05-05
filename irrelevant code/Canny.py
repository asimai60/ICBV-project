import time

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import feature
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter
from skimage.draw import circle_perimeter
import time

plt.close('all')
# Set the directory path
plastic_path = r'C:\Users\nrhot\Downloads\WhatsApp Unknown 2024-04-24 at 12.53.26\PLAST'
glass_path = r'C:\Users\nrhot\Downloads\WhatsApp Unknown 2024-04-24 at 12.53.26\GLASS'
save_directory = r"C:\Users\nrhot\Downloads\WhatsApp Unknown 2024-04-24 at 12.53.26\cropped"
Local_max_Th = 1.2
LOW_threshold = 30
HIGH_threshold = 110
bin_size = 3
radius = 15
K_size = (7, 7)


def load_and_resize(path):
    image = cv2.imread(path)
    factor = int(np.round(max(image.shape[0], image.shape[1]) / 1000))
    desired_shape = (image.shape[1]//factor, image.shape[0]//factor)
    image = cv2.resize(image, desired_shape)
    return image


def canny(gray, ax):
    blurred = cv2.GaussianBlur(gray, K_size, 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    ret2, otsu_thresh = cv2.threshold(adaptive_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(adaptive_thresh, 0.3 * ret2, 1.5 * ret2)

    ax.imshow(edges, cmap='gray')
    ax.set_title('Edge Detection')
    ax.axis('off')
    return edges


def plotCircles(image, detected_circles, bin_size, image_path, ax):
    '''
    This function plots the detected circles and returns a square region around each circle
    with only the circle visible (masked).
    `detected_circles` should be an iterable of tuples (radius, y-coordinate, x-coordinate).
    '''
    ax.imshow(image, cmap="gray")
    ax.title.set_text(image_path.split('\\')[-1])  # Assuming Windows path, use basename to simplify title

    for rad, cy, cx in detected_circles:
        # Calculate the square coordinates based on the bin size and the desired radius.
        top_left_x = max(int(cx * bin_size) - int(rad * bin_size), 0)
        top_left_y = max(int(cy * bin_size) - int(rad * bin_size), 0)
        bottom_right_x = min(int(cx * bin_size) + int(rad * bin_size), image.shape[1])
        bottom_right_y = min(int(cy * bin_size) + int(rad * bin_size), image.shape[0])

        # Crop the image to the square around the detected circle.
        cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x].copy()

        # Create a mask for the cropped image.
        mask = np.zeros_like(cropped_image, dtype=np.uint8)
        circle_center_x = cropped_image.shape[1] // 2
        circle_center_y = cropped_image.shape[0] // 2
        cv2.circle(mask, (circle_center_x, circle_center_y), int(rad * bin_size), (255, 255, 255), -1)

        # Convert mask to grayscale if it's not already.
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Apply the mask to the cropped image.
        masked_circle_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

        # Draw the circle on the original image for visualization.
        cv2.circle(image, (int(cx * bin_size), int(cy * bin_size)), int(rad * bin_size), (0, 255, 0), 2)

    # Show the cropped image with the circle visible in the subplot
    ax.imshow(masked_circle_image, cmap="gray")
    ax.set_title("Cropped Circle")
    ax.axis('off')

    # Assuming `save_directory` is globally accessible
    new_filename = image_path[65:]
    full_path = os.path.join(save_directory, new_filename)

    # Save the cropped image to the specified path
    cv2.imwrite(full_path, masked_circle_image)

    print(f"Saved: {full_path}")
    return masked_circle_image


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


def HoughCircles(image, image_path):
    # Set the radius, cx and cy range
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_map = canny(gray)
    height, width = edge_map.shape
    max_radius = width // 2

    accumulator = generate_accumulator(edge_map, max_radius, bin_size)

    local_maxima = find_global_max(accumulator)
    local_maxima = [local_maxima]

    masked_image = plotCircles(image, local_maxima, bin_size, image_path)
    return masked_image


# List all files in the directory
def main(path):
    files = os.listdir(path)
    # Process each image
def main(path):
    files = os.listdir(path)
    for image_file in files:
        image_path = os.path.join(path, image_file)
        image = load_and_resize(image_path)
        fig, axs = plt.subplots(2, 1, figsize=(12, 6))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_map = canny(gray, axs[0])
        height, width = edge_map.shape
        max_radius = width // 2
        accumulator = generate_accumulator(edge_map, max_radius, bin_size)
        local_maxima = find_global_max(accumulator)
        local_maxima = [local_maxima]

        plotCircles(image, local_maxima, bin_size, image_path, axs[1])
        plt.show()
        time.sleep(2)  # Pause to allow manual inspection of the plots


print("glass data")
main(glass_path)
print("plastic data:")
main(plastic_path)
