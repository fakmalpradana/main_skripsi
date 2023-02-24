from utils.bantu import BacaData, Dataset
from utils.hex2rgb import Label
from matplotlib import pyplot as plt

import numpy as np

img_dataset = BacaData('image').ambilRGB()
msk_dataset = BacaData('mask').ambil()
dem_dataset = BacaData('dem').ambil()

X_data, Y_data = Dataset(img_dataset, msk_dataset, dem_dataset).splitXY()

print(f'Ukuran data input (gambar dan dem) adalah {X_data.shape}')
print(f'Ukuran data label atau data mask adalah {Y_data.shape}')

label = Label(msk_dataset).convert()

print(f'jumlah kelas yg ada : {np.unique(label)}')

# plot img
# image_number = random.randint(0, len(img_dataset))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(img_dataset[8])
# plt.subplot(122)
# plt.imshow(msk_dataset[8])
# plt.show()