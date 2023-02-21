import os
import cv2
import numpy as np

from patchify import patchify
from PIL import Image

# Konstanta
data_dir = 'dataset/'
patch_size = 512

for path, subdirs, files in os.walk(data_dir):
    print(path)