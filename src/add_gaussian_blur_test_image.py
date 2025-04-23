import cv2
import os
import glob as glob

from tqdm import tqdm

src_dir = 'E:\\6th sem\\AI LAB\\AI_labexam\\test_data'
images = glob.glob(f"{src_dir}/*.jpg")
dst_dir = 'E:\\6th sem\\AI LAB\\AI_labexam\\test_data\\gaussian_blurred'

for i, img in tqdm(enumerate(images), total=len(images)):
    img = cv2.imread(f"{images[i]}")
    # add gaussian blurring
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    cv2.imwrite(f"{dst_dir}/image_{i+1}.jpg", blur)

print('DONE')