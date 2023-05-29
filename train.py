# Masteran, edit nya di train 2 aja

from utils.bantu import BacaData, Dataset
from utils.hex2rgb import Label
from utils.model import AkmalCNN
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import numpy as np

# mengambil data dari direktori
img_dataset = BacaData('data/ortho_30.png').patchData(512, 0.625)
msk_dataset = BacaData('data/mask_30.png').patchData(512, 0.625)
hsd_dataset = BacaData('data/hill_30.png').patchData(512, 0.625)
sgm_dataset = BacaData('data/segment_30.png').patchData(512, 0.625)

# preprosesing data (pembuatan dataset, convert label, dan train test split data)
X_data, Y_data = Dataset(img_dataset, msk_dataset, hsd_dataset, sgm_dataset).splitXY()
Y_data = Label(Y_data).convert()

# # percobaan sementara tanpa dem
# X_data = np.array(img_dataset)

Y_data = to_categorical(Y_data, num_classes=len(np.unique(Y_data)))
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 42)

# cek ukuran data
print(f'Ukuran data train adalah {X_train.shape}')
print(f'Ukuran label train adalah {y_train.shape}')
print(f'Ukuran data test adalah {X_test.shape}')
print(f'Ukuran label test adalah {y_test.shape}')

# plt.figure(figsize=(12, 6))
# plt.subplot(131)
# plt.imshow(X_data[25][:,:,6:9])
# plt.subplot(132)
# plt.imshow(X_data[25][:,:,3:6])
# plt.subplot(133)
# plt.imshow(msk_dataset[25])
# plt.show()

# -------------------------------AREA BAHAYA-------------------------------

# hapus variabel yg tidak diperlukan
del X_data, Y_data, img_dataset, msk_dataset, hsd_dataset, sgm_dataset

# buat dan compile model
shape = AkmalCNN(7, 512, 512, 9)
model = shape.buatModel()
compiled_model = shape.compileModel(model)

# training model
compiled_model.fit(
    X_train,
    y_train,
    batch_size=8,
    verbose=1,
    epochs=500,
    validation_data=(X_test, y_test),
    shuffle=False,
)

# save model
compiled_model.save('model/e500_k3_v4.h5')

# 55 commit deleted