import numpy as np
import cv2

def row_by_row_blacking_like_ray_tracking(img):      
    blackened_img = 255 * np.ones_like(img)
    for x in range(2, img.shape[0] - 2):
        black_pxls = np.where(img[x, :] == 0)
        if black_pxls[0].size <= 10: continue
        ystart = black_pxls[0][0]
        yend = black_pxls[0][-1]
        blackened_img[x,ystart:yend+1] = 0
        # cv2.imshow('row_by_row', blackened_img)
        # cv2.waitKey(0)
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

def keep_largest_blob(binary_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8, ltype=cv2.CV_32S)
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    output_image = np.zeros_like(binary_mask)
    output_image[labels == largest_component] = 255

    return output_image

def vertical_filling(img, k=10):
    for y in range(2, img.shape[1] - 2):
        for x in range(2, img.shape[0] - 2):
            if img[x, y] == 0:
                if any(img[x-k:x+1, y] == 255) and any(img[x+1:x+k, y] == 255):
                    img[x, y] = 255
    return img


def vertical_scanning(img, threshold=100):
    stop = False
    for y in range(20, img.shape[1] - 20):
        if stop:
            break
        num_black = img.shape[0] - cv2.countNonZero(img[:,y])
        if num_black <=threshold:
            img[:,:y+1] = 255
        elif num_black > threshold:
            stop = True
    stop = False
    for y in range(img.shape[1] - 20, 20, -1):
        if stop:
            break
        num_black = img.shape[0] - cv2.countNonZero(img[:,y])
        if num_black <= threshold:
            img[:,y:] = 255
        elif num_black > threshold:
            stop = True
    return img

def std_vertical_scanning(img, threshold=100):
    for y in range(2, img.shape[1] - 2):
        black_pxls = np.where(img[:,y] == 0)[0]
        if len(black_pxls) > 0:
            gaps = np.diff(black_pxls)
            if (np.max(gaps) > threshold) or len(black_pxls) < 60:
                img[:,y] = 255
    cv2.imshow('std_vertical_scanning', img)
    cv2.waitKey(0)
    return img

def black_if_many_switches(img):
    for y in range(2, img.shape[1] - 2):
        switch_count = 0
        for x in range(2, img.shape[0] - 2):
            if img[x,y] != img[x-1,y]:
                switch_count += 1
            if switch_count > 4:
                img[:,y] = 0
                break
    return img


for i in range(2, 20):
    path = 'bottles/' + str(i) + '.jpeg'
    image = load_and_resize(path)
    erosed_image = generate_erosion_image(image)
    erosed_image[:,:100] = 255
    erosed_image[:,-100:] = 255
    cv2.imshow('erosed_image', erosed_image)
    erosed_image = cv2.blur(erosed_image, (20,30))
    cv2.imshow('avarage_image', erosed_image)
    erosed_image = cv2.threshold(erosed_image, 127, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('binary_image', erosed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    row_by_row_mask = cv2.bitwise_not(row_by_row_blacking_like_ray_tracking(erosed_image))
    # row_by_row_mask = keep_largest_blob(row_by_row_mask)
    row_by_row_mask = vertical_filling(row_by_row_mask)
    segmented_row_by_row = cv2.bitwise_and(image,image,mask=row_by_row_mask)
    
    cv2.imshow('segmented', segmented_row_by_row)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

exit()
path = 'bottles/17.jpeg'
image = load_and_resize(path)
erosed_image = generate_erosion_image(image)

row_by_row_mask = cv2.bitwise_not(row_by_row_blacking_like_ray_tracking(erosed_image))
segmented_row_by_row = cv2.bitwise_and(image,image,mask=row_by_row_mask)
cv2.imshow('Original Image', image)
cv2.imshow('segmented', segmented_row_by_row)
cv2.waitKey(0)

