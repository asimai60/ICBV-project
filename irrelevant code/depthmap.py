import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess_image(image, target_size=384):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def postprocess_depth_map(depth):
    depth = depth.cpu().squeeze().numpy()
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth = np.uint8(depth)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
    return depth

def main(image_path):
    # Load and preprocess the image
    image = load_image(image_path)
    input_tensor = preprocess_image(image)

    # Load MiDaS model
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    midas.to(device)
    midas.eval()

    # Predict depth
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        depth_prediction = midas(input_tensor)

    # Postprocess and save the depth map
    depth_image = postprocess_depth_map(depth_prediction)
    # cv2.imwrite('depth_map.png', depth_image)
    cv2.imshow('Depth Map', depth_image)

if __name__ == "__main__":
    import os
    for im in os.listdir('bottles'):
        main(f'bottles/{im}')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # main("bottle pictures/b.jpeg")
