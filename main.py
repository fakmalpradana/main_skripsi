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
aoi = BacaData('data/ortho_ugm_65.png').patchData(512, 1)
hil = BacaData('data/hill_ugm_65.png').patchData(512, 1)
seg = BacaData('data/sgm_ugm_65.png').patchData(512, 1)
msk = BacaData('data/mask_ugm_65.png').patchData(512, 1)

# pre-prosesing
X_data, Y_data = Dataset(aoi, msk, hil, seg).splitXY()
Y_data = Label(Y_data).convert()
Y_data = to_categorical(Y_data, num_classes=len(np.unique(Y_data)))

y_argmax = np.argmax(Y_data, axis=3)

# proses klasifikasi
predict = []
for i in range(X_data.shape[0]):
    test_img = X_data[i]
    test_img_input = np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]

    predict.append(predicted_img)

predict = np.array(predict)

print(predict.shape) # isinya (63, 512, 512)

size = (7, 8, 1, 512, 512, 1)

# rebuild patch gambar
unpatch_img = predict.reshape(size)
print(unpatch_img.shape)

reconstructed_image = p.unpatchify(unpatch_img, (3584, 4096, 1))
print(f'ukuran akhir prediksi {reconstructed_image.shape}')

out_img = reconstructed_image[:3227, :3899, :]
out_img = cv2.merge((out_img, out_img, out_img))
print(f'ukuran akhir output {out_img.shape}')

out_img = Label(out_img).invert()
out_img = out_img.astype(np.uint8)

# predicted_img = np.argmax(reconstructed_image, axis=0)[0,:,:]
# plt.imshow(reconstructed_image)
# plt.show()
# print(np.unique(predicted_img))
cv2.imwrite('out/ugm_new.png', out_img)