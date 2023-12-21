import numpy as np
import skimage
import deskew
import os
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

def align_image(image):
    image[image < 255] += 1
    grayscale = skimage.color.rgb2gray(image)
    angle = deskew.determine_skew(grayscale)
    rotated = skimage.transform.rotate(image, angle, resize=True)*255
    rotated[rotated == 0] = 255
    rotated[rotated > 0] -= 1

    return rotated


directory = '../../Загрузки/many_tables/'
aligned_directory = '../../Загрузки/aligned_many_tables/'
image_files = [file for file in os.listdir(directory) if file.endswith('.png')]

for image_file in tqdm(image_files, desc='Выравнивание изображений'):
    
    image_path = os.path.join(directory, image_file)
    image = skimage.io.imread(image_path)
    
    aligned_image = align_image(image)
    
    aligned_image_path = os.path.join(aligned_directory, image_file)
    cv2.imwrite(aligned_image_path, aligned_image)



f, ax = plt.subplots(1,2, figsize=(15,9))

ax[0].imshow(image)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('Original')

ax[1].imshow(aligned_image/255)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Rotated')

plt.show()
