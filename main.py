from utils.bantu import BacaData, Dataset
from utils.hex2rgb import Label
from matplotlib import pyplot as plt

import numpy as np
import random

img_dataset = BacaData('image').ambilRGB()
msk_dataset = BacaData('mask').ambilRGB()
dem_dataset = BacaData('dem').ambil()

X_data, Y_data = Dataset(img_dataset, msk_dataset, dem_dataset).splitXY()
Y_data = Label(Y_data).convert()

print(f'Ukuran data input (gambar dan dem) adalah {X_data.shape}')
print(f'Ukuran data label atau data mask adalah {Y_data.shape}')
print(f'jumlah kelas yg ada : {np.unique(Y_data)}')

# plot img
image_number = random.randint(0, len(img_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img_dataset[image_number])
plt.subplot(122)
plt.imshow(Y_data[image_number][:,:,0])
plt.show()