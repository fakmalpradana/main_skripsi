from keras.models import load_model
from utils.model import AkmalCNN
from utils.bantu import BacaData, Dataset
from utils.hex2rgb import Label
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

import numpy as np
import patchify as p
import cv2

scaler = MinMaxScaler()
model = load_model('model/e500_k3_v4.h5',)

# plot_model(model, "model_3k.png", show_shapes=True)

aoi = cv2.imread('data/ortho_30.png', 1)
hil = cv2.imread('data/hill_30.png', 1)
seg = cv2.imread('data/segment_aoi.png', 1)

patch_size = 512

def arr2img(img):
    SIZE_X = (img.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
    SIZE_Y = (img.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
    large_img = Image.fromarray(img)
    large_img = large_img.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
    #image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
    large_img = np.array(large_img)

    return large_img

large_img = np.concatenate((arr2img(hil), arr2img(aoi), arr2img(seg)), axis=-1)

patches_img = p.patchify(large_img, (512, 512, 9), step=448)  #Step=256 for 256 patches means no overlap
patches_img = patches_img[:,:,0,:,:,:]

patched_prediction = []
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        
        single_patch_img = patches_img[i,j,:,:,:]
        
        #Use minmaxscaler instead of just dividing by 255. 
        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        single_patch_img = np.expand_dims(single_patch_img, axis=0)
        pred = model.predict(single_patch_img)
        pred = np.argmax(pred, axis=3)
        pred = pred[0,:,:]
                                 
        patched_prediction.append(pred)

patched_prediction = np.array(patched_prediction)
patched_prediction = np.reshape(patched_prediction, [patches_img.shape[0], patches_img.shape[1], patches_img.shape[2], patches_img.shape[3]])

# img_x, img_y = 4365, 3304
unpatched_prediction = p.unpatchify(patched_prediction, (4365, 3304))

print('large_img : ',large_img.shape)
print('patches_img : ', patches_img.shape)
print('patched_prediction : ', patched_prediction.shape)
print('unpatched_prediction : ', unpatched_prediction.shape)
# gabung seluruh data hasil prediksi
plt.imshow(unpatched_prediction)
plt.axis('off')