import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2
import models
import torch

from torchvision.transforms import transforms
from torchvision.utils import save_image

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

device = 'cpu'

# load the trained model
model = models.CNN().to(device).eval()
model.load_state_dict(torch.load('E:\\6th sem\\AI LAB\\AI_labexam\\outputs\\model.pth'))

# define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

name = 'image_1'

image = cv2.imread(f"E:\\6th sem\\AI LAB\\AI_labexam\\test_data\\gaussian_blurred{name}.jpg")
orig_image = image.copy()
orig_image = cv2.resize(orig_image, (224, 224))
cv2.imwrite(f"E:\\6th sem\\AI LAB\\AI_labexam\\outputs\\test_deblurred_images\\original_blurred_image_2.jpg", orig_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = transform(image).unsqueeze(0)
print(image.shape)

with torch.no_grad():
    outputs = model(image)
    save_decoded_image(outputs.cpu().data, name=f"E:\\6th sem\\AI LAB\\AI_labexam\\outputs\\test_deblurred_images\\original_blurred_image_2.jpg")

