import numpy as np
import cv2
import seaborn as sns

from matplotlib import pyplot as plt
from utils.hex2rgb import Label
from sklearn import metrics as m
from imblearn import metrics as mm

x = cv2.imread('out/ugm_35.png')
y = cv2.imread('data/mask_ugm_65.png')

x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)

pred = np.expand_dims(x, axis=0)
act = np.expand_dims(y, axis=0)

prediction = Label(pred).convert()
actual = Label(act).convert()

cm = m.confusion_matrix(actual.flatten(), prediction.flatten(), labels=[1, 2, 3, 4, 5, 6])
acc = m.accuracy_score(actual.flatten(), prediction.flatten())
rec = mm.sensitivity_score(actual.flatten(), prediction.flatten(), average='macro')
pre = m.precision_score(actual.flatten(), prediction.flatten(), average='weighted')
spe = mm.specificity_score(actual.flatten(), prediction.flatten(), average='weighted')

print(f'accuracy : {acc}')
print(f'recall : {rec}')
print(f'precision : {pre}')
print(f'sensitivity : {spe}')

# # plot matrix konfusi
# label = ['Sawah', 'Sungai', 'Pepohonan', 'Jalan', 'Ground/RTH', 'Bangunan']

# plt.figure(figsize=(8,6), dpi=100)

# ax = sns.heatmap(
#     100*cm/np.sum(cm), 
#     annot=True, 
#     vmax=20, 
#     vmin=0, 
#     cmap='viridis',
#     xticklabels=label,
#     yticklabels=label,
#     )

# ax.set_yticklabels(ax.get_yticklabels(), rotation=60)
# ax.set(xlabel='Hasil klasifikasi', ylabel='Data sebenarnya')
# ax.set_title('Matrix Konfusi')

# plt.show()