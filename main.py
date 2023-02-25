from utils.bantu import BacaData, Dataset
from utils.hex2rgb import Label
from utils.model import AkmalCNN
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import numpy as np
import random

# mengambil data dari direktori
img_dataset = BacaData('image').ambilRGB()
msk_dataset = BacaData('mask').ambilRGB()
dem_dataset = BacaData('dem').ambil()

# preprosesing data (pembuatan dataset, convert label, dan train test split data)
X_data, Y_data = Dataset(img_dataset, msk_dataset, dem_dataset).splitXY()
Y_data = Label(Y_data).convert()

Y_data = to_categorical(Y_data, num_classes=len(np.unique(Y_data)))
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size = 0.40, random_state = 42)

# cek ukuran data
print(f'Ukuran data train adalah {X_train.shape}')
print(f'Ukuran label train adalah {y_train.shape}')
print(f'Ukuran data test adalah {X_test.shape}')
print(f'Ukuran label test adalah {y_test.shape}')

# shape = AkmalCNN(7, 512, 512, 4)
# model = shape.buatModel()
# compiled_model = shape.compileModel(model)