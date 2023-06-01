from keras.models import load_model
from utils.bantu import BacaData, Dataset
from utils.hex2rgb import Label
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from utils.hex2rgb import Label

import patchify as p
import numpy as np
import cv2

# load model
scaler = MinMaxScaler()
model = load_model('model/e250_v4_35.h5',)

# load data
aoi = BacaData('data/klaten4/ortho4.png').patchData(512, 1)
hil = BacaData('data/klaten4/hill4.png').patchData(512, 1)
seg = BacaData('data/klaten4/sgm2.png').patchData(512, 1)
msk = BacaData('data/klaten4/sgm2.png').patchData(512, 1)

# pre-prosesing
X_data, Y_data = Dataset(aoi, msk, hil, seg).splitXY()

# proses klasifikasi
predict = []
for i in range(X_data.shape[0]):
    test_img = X_data[i]
    test_img_input = np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]

    predict.append(predicted_img)

predict = np.array(predict)

print(predict.shape) # isinya (2, 512, 512)

# rebuild patch gambar
size = (5, 5, 1, 512, 512, 1)

unpatch_img = predict.reshape(size)
print(unpatch_img.shape)

reconstructed_image = p.unpatchify(unpatch_img, (2560, 2560, 1))
print(f'ukuran akhir prediksi {reconstructed_image.shape}')

out_img = reconstructed_image[:2449, :2067, :]
out_img = cv2.merge((out_img, out_img, out_img))
print(f'ukuran akhir output {out_img.shape}')

out_img = Label(out_img).invert()
out_img = out_img.astype(np.uint8)

cv2.imwrite('out/newdata_004.png', out_img)

# # coba plot
# test_img_number = 0
# test_img = X_data[test_img_number]

# test_img_input=np.expand_dims(test_img, 0)
# prediction = (model.predict(test_img_input))
# predicted_img=np.argmax(prediction, axis=3)[0,:,:]

# plt.figure(figsize=(12, 8))
# plt.subplot(221)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,3:6])

# plt.subplot(222)
# plt.title('Prediction on test image')
# plt.imshow(predicted_img)
# plt.show()