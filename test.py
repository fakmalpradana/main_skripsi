from keras.models import load_model
from utils.model import AkmalCNN
from utils.bantu import BacaData, Dataset
from utils.hex2rgb import Label
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import numpy as np
import random

model = load_model(
    'model/test_model.h5',
    custom_objects={
        'dice_loss_plus_1focal_loss': AkmalCNN(7, 512, 512, 9).bobot(),
        'jacard_coef': AkmalCNN(7, 512, 512, 9).jacard_coef
    })

# mengambil data dari direktori
img_dataset = BacaData('image').ambilRGB()
msk_dataset = BacaData('mask').ambilRGB()
hsd_dataset = BacaData('hsd').ambilRGB()
sgm_dataset = BacaData('sgm').ambilRGB()

# preprosesing data (pembuatan dataset, convert label, dan train test split data)
X_data, Y_data = Dataset(img_dataset, msk_dataset, hsd_dataset, sgm_dataset).splitXY()
Y_data = Label(Y_data).convert()

# # percobaan sementara tanpa dem
# X_data = np.array(img_dataset)

Y_data = to_categorical(Y_data, num_classes=len(np.unique(Y_data)))
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size = 0.20, random_state = 42)

# test model
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)

test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth = y_test_argmax[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input = np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0,:,:]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,3:6])
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()