import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from glob import glob

# Konstanta
img_dir = 'dataset/*/images/*.png'
patch_size = 512

# Membuat string untuk memanggil dataset
images = []
masks = []
dems = []

for i in range(9):
    images.append('image_{}.png'.format(i))
    masks.append('mask_{}.png'.format(i))
    dems.append('dem_{}.tif'.format(i))

# membaca dataset dan memasukkannya ke dalam variabel
image_dataset = [
    cv2.imread(img) for img in glob(img_dir)
]

image_dataset = [
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_dataset
]

plt.imshow(image_dataset[1])
plt.show()