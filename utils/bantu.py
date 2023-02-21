import cv2
import numpy as np
from glob import glob

# Konstanta
img_dir = 'dataset/*/images/*.png'
patch_size = 512

# Membuat class Data
class Dirname:
    def __init__(self, item, format):
        self.item = item
        self.format = format

    def list(self):
        dir = []
        item = self.item
        format = self.format
        for i in range(9):
            dir.append(f'{item}_{i}.{format}')
        
        return dir

a = Dirname('image', 'png').list()
print(a)

# membaca dataset dan memasukkannya ke dalam variabel
# class Dataset:
#     def __init__(self):
#         self

# image_dataset = [
#     cv2.imread(img) for img in glob(img_dir)
# ]

# image_dataset = [
#     cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_dataset
# ]
