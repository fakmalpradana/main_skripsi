from utils.bantu import BacaData, Dataset
from utils.hex2rgb import Label
from utils.model import AkmalCNN
from matplotlib import pyplot as plt
from keras.utils import to_categorical
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

# percobaan sementara tanpa dem
X_data = np.array(img_dataset)

Y_data = to_categorical(Y_data, num_classes=len(np.unique(Y_data)))
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size = 0.20, random_state = 42)

# cek ukuran data
print(f'Ukuran data train adalah {X_train.shape}')
print(f'Ukuran label train adalah {y_train.shape}')
print(f'Ukuran data test adalah {X_test.shape}')
print(f'Ukuran label test adalah {y_test.shape}')

# # buat dan compile model
# shape = AkmalCNN(7, 512, 512, 3)
# model = shape.buatModel()
# compiled_model = shape.compileModel(model)

# # training model
# compiled_model.fit(
#     X_train,
#     y_train,
#     batch_size=8,
#     verbose=1,
#     epochs=50,
#     validation_data=(X_test, y_test),
#     shuffle=False,
# )

# # save model
# compiled_model.save('model/test_model.h5')

# # test model
# y_pred=compiled_model.predict(X_test)
# y_pred_argmax=np.argmax(y_pred, axis=3)
# y_test_argmax=np.argmax(y_test, axis=3)

# test_img_number = random.randint(0, len(X_test))
# test_img = X_test[test_img_number]
# ground_truth=y_test_argmax[test_img_number]
# #test_img_norm=test_img[:,:,0][:,:,None]
# test_img_input=np.expand_dims(test_img, 0)
# prediction = (compiled_model.predict(test_img_input))
# predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(X_test[2])
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(y_train[2])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(X_train[2])
plt.show()