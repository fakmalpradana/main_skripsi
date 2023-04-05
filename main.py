from keras.models import load_model
from utils.bantu import BacaData, Dataset
from utils.hex2rgb import Label
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import numpy as np

scaler = MinMaxScaler()
model = load_model('model/e500_k3_v4.h5',)

aoi = BacaData('data/ortho_30.png').patchData(512)
hil = BacaData('data/hill_30.png').patchData(512)
seg = BacaData('data/segment_30.png').patchData(512)
msk = BacaData('data/mask_30.png').patchData(512)

X_data, Y_data = Dataset(aoi, msk, hil, seg).splitXY()
Y_data = Label(Y_data).convert()
Y_data = to_categorical(Y_data, num_classes=len(np.unique(Y_data)))

y_argmax = np.argmax(Y_data, axis=3)

predict = []
for i in range(X_data.shape[0]):
    test_img = X_data[i]
    test_img_input = np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]

    predict.append(predicted_img)

predict = np.array(predict)

print(predict.shape) # isinya (130, 512, 512)