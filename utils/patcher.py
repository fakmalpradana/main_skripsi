import numpy as np
import patchify as p
import cv2
from matplotlib import pyplot as plt

# bagian ambil gambar
segment = cv2.imread('/home/fairuzakmal/skripsiCNN/data/segment_30.png')
segment = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
segment = np.array(segment)
size_ori = segment.shape  # shape 9901, 13080, 3

# bagian menentukan ukuran channel dan pembulatan ke atas
ch1 = np.ceil(segment.shape[0]/512).astype(int)
ch2 = np.ceil(segment.shape[1]/512).astype(int)

# menambahkan kolom dan baris kosong pd gambar
arr0 = np.zeros((ch1*512, ch2*512, 3))
arr0[:segment.shape[0], :segment.shape[1]] += segment
arr = arr0.astype(np.uint8)

print(f'size awal : {segment.shape}')
print(f'size akhir : {arr.shape}')

print(f'channel 1 : {ch1}')
print(f'channel 2 : {ch2}')

# bagian patch gambar
patches = p.patchify(arr, (512, 512, 3), step=512)
print(patches.shape)

img_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        img_patches.append(patches[i,j,0,:,:])

img_patches = np.array(img_patches)

print(img_patches.shape)

# bagian unpatch gambar
unpatch_img = img_patches.reshape(patches.shape)
print(unpatch_img.shape)

reconstructed_image = p.unpatchify(patches, arr.shape)

plt.imshow(reconstructed_image)
plt.show()
