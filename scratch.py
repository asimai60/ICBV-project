import numpy as np
import cv2
from skimage import feature
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def row_by_row_blacking_like_ray_tracking(img):
    blackened_img = 255 * np.ones_like(img)
    for x in range(2, img.shape[0] - 2):
        black_pxls = np.where(img[x, :] == 0)
        if black_pxls[0].size == 0: continue
        ystart = black_pxls[0][0]
        yend = black_pxls[0][-1]
        blackened_img[x,ystart:yend+1] = 0
    #cv2.imshow('row_by_row', blackened_img)
    cv2.waitKey(0)
    return blackened_img

def load_and_resize(path):
    image = cv2.imread(path)
    factor = int(np.round(max(image.shape[0], image.shape[1]) / 1000))
    desired_shape = (image.shape[1]//factor, image.shape[0]//factor)
    image = cv2.resize(image, desired_shape)
    return image

def generate_erosion_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    binary_image = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(binary_image, kernel, iterations=1)
    return erosion


def final(path):
    image = load_and_resize(path)
    erosed_image = generate_erosion_image(image)
    row_by_row_mask = cv2.bitwise_not(row_by_row_blacking_like_ray_tracking(erosed_image))
    segmented_row_by_row = cv2.bitwise_and(image, image, mask=row_by_row_mask)
    #cv2.imshow('Original Image', image)
    #cv2.imshow('segmented', segmented_row_by_row)
    cv2.waitKey(0)
    return segmented_row_by_row


def feature_extraction(image):
    # Load the image and convert it to grayscale
    gray_image = rgb2gray(image)  # Convert to grayscale
    # Convert gray image to uint8
    gray_image = (gray_image * 255).astype(np.uint8)
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

def LBP(image):
    image = rgb2gray(image)
    # Parameters for LBP
    radius = 3
    n_points = 8 * radius

    # Compute the LBP representation
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')

    # Optionally, you can calculate a histogram of the LBP results to use as features
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize the histogram

    print(lbp_hist[0])
    return lbp_hist[0]

paths = [
    r'C:\Users\user\Downloads\bottles_to_extract\12.jpeg',
    r'C:\Users\user\Downloads\bottles_to_extract\13.jpeg',
    r'C:\Users\user\Downloads\bottles_to_extract\17.jpeg',
    r'C:\Users\user\Downloads\bottles_to_extract\16.jpeg',
    r'C:\Users\user\Downloads\bottles_to_extract\3.jpeg',
    r'C:\Users\user\Downloads\bottles_to_extract\4.jpeg',
    r'C:\Users\user\Downloads\bottles_to_extract\5.jpeg',
    r'C:\Users\user\Downloads\bottles_to_extract\18.jpeg',
]

pathspla = [
    r'C:\Users\user\Downloads\bottles_to_extract\plastic\10.jpeg',
    r'C:\Users\user\Downloads\bottles_to_extract\plastic\14.jpeg',
    r'C:\Users\user\Downloads\bottles_to_extract\plastic\19.jpeg',

]
data = []
print("glasses")
# Now iterate over the list of paths directly
for path in paths:
    data_tmp = []
    image = final(path)
    print("feature_extraction")
    feuture = feature_extraction(image)
    print("LBP")
    lbp = LBP(image)
    data_tmp.append(lbp)
    # Calculate the mean brightness and contrast
    mean_brightness = np.mean(image)
    std_contrast = np.std(image)
    data_tmp.append(mean_brightness)
    data_tmp.append(std_contrast)
    data.append(data_tmp)
    print("data" , data_tmp)



    print(f'Mean Brightness: {mean_brightness}')
    print(f'Standard Deviation (Contrast): {std_contrast}')

print("plastic")

for path in pathspla:
    data_tmp = []
    image = final(path)
    print("feature_extraction")
    feuture = feature_extraction(image)
    print("LBP")
    lbp = LBP(image)
    data_tmp.append(lbp)
    # Calculate the mean brightness and contrast
    mean_brightness = np.mean(image)
    std_contrast = np.std(image)
    data_tmp.append(mean_brightness)
    data_tmp.append(std_contrast)
    data.append(data_tmp)
    print("data" , data_tmp)

    print(f'Mean Brightness: {mean_brightness}')
    print(f'Standard Deviation (Contrast): {std_contrast}')


print(data[0])
pca_features = data

# It is a good practice to standardize the features (zero mean and unit variance) before applying PCA
"""scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Initialize PCA - let's say we want to reduce our features to 10 principal components
pca = PCA(n_components=10)

# Fit PCA on the standardized features and transform the data
pca_features = pca.fit_transform(features_standardized)

# Now `pca_features` is the dataset transformed down to 10 dimensions

# To see how much variance is retained after PCA you can check
print(f"Cumulative variance explained by 10 principal components: {np.sum(pca.explained_variance_ratio_):.2f}")"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pca_features = np.array(pca_features)
print("pca_features", pca_features)

# Third principal component

labels = np.array([0, 0, 0, 0,0,0,0,0, 1, 1, 1])  # Binary labels for simplicity

# Create a color map based on labels
colors = ['red' if label == 0 else 'blue' for label in labels]  # Red for label 0, blue for label 1

# Create a new figure for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract individual components
x = pca_features[:, 0]
y = pca_features[:, 1]
z = pca_features[:, 2]

# Plotting data, colored by label
scatter = ax.scatter(x, y, z, c=colors, marker='o')

# Create a legend (optional)
red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Label 0')
blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Label 1')
plt.legend(handles=[red_patch, blue_patch])

# Title and show
plt.title('3D PCA Results')
plt.show()
