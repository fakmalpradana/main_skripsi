import numpy as np
import cv2
import seaborn as sns

from matplotlib import pyplot as plt
from utils.hex2rgb import Label
# from keras.utils import to_categorical
# from sklearn.metrics import confusion_matrix

x = cv2.imread('out/ugm_35.png')
y = cv2.imread('data/mask_ugm_65.png')

x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)

pred = np.expand_dims(x, axis=0)
act = np.expand_dims(y, axis=0)

prediction = Label(pred).convert()
actual = Label(act).convert()

val = np.squeeze(prediction - actual)

acc = (val.size - np.count_nonzero(val)) / val.size
print(f'accuracy : {acc}')

# y_pred = to_categorical(prediction, num_classes=len(np.unique(actual)))
# y_test = to_categorical(actual, num_classes=len(np.unique(actual)))

# cm = confusion_matrix(y_test[:, -1].astype(np.uint8), y_pred[:, -1].astype(np.uint8))

# FP = len(np.where(prediction - actual == 1)[0])
# FN = len(np.where(prediction - actual == -1)[0])
# TP = len(np.where(prediction + actual == 2)[0])
# TN = len(np.where(prediction + actual == 0)[0])

# Aa = len(np.where((prediction - actual == 0) & (actual == 1)))
# Ab = len(np.where((prediction - actual == 1) & (actual == 1)))
# Ac = len(np.where((prediction - actual == 2) & (actual == 1)))
# Ad = len(np.where((prediction - actual == 3) & (actual == 1)))
# Ae = len(np.where((prediction - actual == 4) & (actual == 1)))
# Af = len(np.where((prediction - actual == 5) & (actual == 1)))

# Ba = len(np.where((prediction - actual == -1) & (actual == 2)))
# Bb = len(np.where((prediction - actual == 0) & (actual == 2)))
# Bc = len(np.where((prediction - actual == 1) & (actual == 2)))
# Bd = len(np.where((prediction - actual == 2) & (actual == 2)))
# Be = len(np.where((prediction - actual == 3) & (actual == 2)))
# Bf = len(np.where((prediction - actual == 4) & (actual == 2)))

# Ca = len(np.where((prediction - actual == -2) & (actual == 3)))
# Cb = len(np.where((prediction - actual == -1) & (actual == 3)))
# Cc = len(np.where((prediction - actual == 0) & (actual == 3)))
# Cd = len(np.where((prediction - actual == 1) & (actual == 3)))
# Ce = len(np.where((prediction - actual == 2) & (actual == 3)))
# Cf = len(np.where((prediction - actual == 3) & (actual == 3)))

# Da = len(np.where((prediction - actual == -3) & (actual == 4)))
# Db = len(np.where((prediction - actual == -2) & (actual == 4)))
# Dc = len(np.where((prediction - actual == -1) & (actual == 4)))
# Dd = len(np.where((prediction - actual == 0) & (actual == 4)))
# De = len(np.where((prediction - actual == 1) & (actual == 4)))
# Df = len(np.where((prediction - actual == 2) & (actual == 4)))

# Ea = len(np.where((prediction - actual == -4) & (actual == 5)))
# Eb = len(np.where((prediction - actual == -3) & (actual == 5)))
# Ec = len(np.where((prediction - actual == -2) & (actual == 5)))
# Ed = len(np.where((prediction - actual == -1) & (actual == 5)))
# Ee = len(np.where((prediction - actual == 0) & (actual == 5)))
# Ef = len(np.where((prediction - actual == 1) & (actual == 5)))

# Fa = len(np.where((prediction - actual == -5) & (actual == 6)))
# Fb = len(np.where((prediction - actual == -4) & (actual == 6)))
# Fc = len(np.where((prediction - actual == -3) & (actual == 6)))
# Fd = len(np.where((prediction - actual == -2) & (actual == 6)))
# Fe = len(np.where((prediction - actual == -1) & (actual == 6)))
# Ff = len(np.where((prediction - actual == 0) & (actual == 6)))

# cmat = [
#     [Aa, Ab, Ac, Ad, Ae, Af],
#     [Ba, Bb, Bc, Bd, Be, Bf],
#     [Ca, Cb, Cc, Cd, Ce, Cf],
#     [Da, Db, Dc, Dd, De, Df],
#     [Ea, Eb, Ec, Ed, Ee, Ef],
#     [Fa, Fb, Fc, Fd, Fe, Ff]
# ]
# acc = (Aa + Bb + Cc + Dd + Ee + Ff)/(Ab + Ac + Ad + Ae + Af + Ba + Bc + Bd + Be + Bf + Ca + Cb + Cd + Ce + Cf + Da + Db + Dc + De + Df + Ea + Eb + Ec + Ed + Ef + Fa + Fb + Fc + Fd + Fe)
# print(f'accuracy : {Fe}')

# cmat = [[TP, FN], [FP, TN]]
# acc = (TP+TN)/(TP+FP+TN+FN)
# print(f'accuracy : {acc}')

# plt.figure(figsize = (6,6))
# sns.heatmap(cmat/np.sum(cmat), cmap="Reds", annot=True, fmt = '.2%', square=1,   linewidth=2.)
# plt.xlabel("predictions")
# plt.ylabel("real values")
# plt.show()
