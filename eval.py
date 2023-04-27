import numpy as np
import cv2
import seaborn as sns

from matplotlib import pyplot as plt
from utils.hex2rgb import Label

pred = np.expand_dims(cv2.imread('out/ugm_25.png'), axis=0)
act = np.expand_dims(cv2.imread('data/mask_ugm_65.png'), axis=0)

prediction = Label(pred).convert()
actual = Label(act).convert()

FP = len(np.where(prediction - actual == 1)[0])
FN = len(np.where(prediction - actual == -1)[0])
TP = len(np.where(prediction + actual == 2)[0])
TN = len(np.where(prediction + actual == 0)[0])

cmat = [[TP, FN], [FP, TN]]
acc = (TP+TN)/(TP+FP+TN+FN)
print(f'accuracy : {acc}')

plt.figure(figsize = (6,6))
sns.heatmap(cmat/np.sum(cmat), cmap="Reds", annot=True, fmt = '.2%', square=1,   linewidth=2.)
plt.xlabel("predictions")
plt.ylabel("real values")
plt.show()
