from keras.models import load_model
from utils.model import AkmalCNN
from utils.bantu import BacaData, Dataset
from utils.hex2rgb import Label
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import patchify as p
import random

scaler = MinMaxScaler()
model = load_model(
    'model/e500_k3_v2.h5',
    custom_objects={
        'dice_loss_plus_1focal_loss': AkmalCNN(7, 512, 512, 9).bobot(),
        'jacard_coef': AkmalCNN(7, 512, 512, 9).jacard_coef
    })

plot_model(model, "model_3k.png", show_shapes=True)

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

# Y_data = to_categorical(Y_data, num_classes=len(np.unique(Y_data)))
# X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size = 0.50, random_state = 42)

# test model
# y_pred = model.predict(X_data) # shape 63, 512, 512, 7
# y_pred_argmax = np.argmax(y_pred, axis=3)
# y_test_argmax = np.argmax(Y_data, axis=3)

# test_img_number = random.randint(0, len(X_data)-1)
# test_img = X_data[test_img_number]
# ground_truth = y_test_argmax[test_img_number]
# #test_img_norm=test_img[:,:,0][:,:,None]
# test_img_input = np.expand_dims(test_img, 0)
# prediction = (model.predict(test_img_input))
# predicted_img = np.argmax(prediction, axis=3)[0,:,:]

# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,3:6])
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth)
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(predicted_img)
# plt.show()

patched_prediction = []
for i in range(X_data.shape[0]):
    for j in range(X_data.shape[1]):
        
        single_patch_img = X_data[i,j,:,:,:]
        
        #Use minmaxscaler instead of just dividing by 255. 
        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        single_patch_img = np.expand_dims(single_patch_img, axis=0)
        pred = model.predict(single_patch_img)
        pred = np.argmax(pred, axis=3)
        pred = pred[0, :,:]
                                 
        patched_prediction.append(pred)

patched_prediction = np.array(patched_prediction)
patched_prediction = np.reshape(patched_prediction, [X_data.shape[0], X_data.shape[1], X_data.shape[2], X_data.shape[3]])

img_x, img_y = 4365, 3304
unpatched_prediction = p.unpatchify(patched_prediction, (img_x, img_y))

# gabung seluruh data hasil prediksi
# plt.imshow(pred)
# plt.show()