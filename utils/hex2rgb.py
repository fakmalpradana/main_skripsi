import numpy as np
import cv2

class Label:
    def __init__(self, mask):
        self.mask = mask

    def hex2rgb(self, label):
        BANGUNAN = '#69008C'.lstrip('#')
        BANGUNAN = np.array(tuple(int(BANGUNAN[i:i+2], 16) for i in (0, 2, 4))) # 105 0 140

        JALAN = '#FF0000'.lstrip('#')
        JALAN = np.array(tuple(int(JALAN[i:i+2], 16) for i in (0, 2, 4))) # 255 0 0

        SAWAH = '#00FF00'.lstrip('#') 
        SAWAH = np.array(tuple(int(SAWAH[i:i+2], 16) for i in (0, 2, 4))) # 0 255 0

        SUNGAI =  '529C9A'.lstrip('#') 
        SUNGAI = np.array(tuple(int(SUNGAI[i:i+2], 16) for i in (0, 2, 4))) # 82 156 154

        PEPOHONAN = '004B00'.lstrip('#') 
        PEPOHONAN = np.array(tuple(int(PEPOHONAN[i:i+2], 16) for i in (0, 2, 4))) # 0 75 0

        RTH = '#BF9C00'.lstrip('#') 
        RTH = np.array(tuple(int(RTH[i:i+2], 16) for i in (0, 2, 4))) # 191 156 0

        label_seg = np.zeros(label.shape,dtype=np.uint8)
        label_seg [np.all(label == BANGUNAN,axis=-1)] = 6
        label_seg [np.all(label == SAWAH,axis=-1)] = 1
        label_seg [np.all(label == SUNGAI,axis=-1)] = 2
        label_seg [np.all(label == PEPOHONAN,axis=-1)] = 3
        label_seg [np.all(label == JALAN,axis=-1)] = 4
        label_seg [np.all(label == RTH,axis=-1)] = 5

        label_seg = label_seg[:,:,0]

        return label_seg

    def convert(self):
        labels = []
        mask = np.array(self.mask)
        for i in range(mask.shape[0]):
            label = self.hex2rgb(mask[i])
            labels.append(label)

        return np.expand_dims(np.array(labels), axis=3)
    
    def invert(self):
        mask = self.mask
        b, g, r = cv2.split(mask)

        r[r==1] = 105
        r[r==2] = 255
        r[r==3] = 0
        r[r==4] = 82
        r[r==5] = 0
        r[r==6] = 191

        g[g==1] = 0
        g[g==2] = 0
        g[g==3] = 255
        g[g==4] = 156
        g[g==5] = 75
        g[g==6] = 156

        b[b==1] = 140
        b[b==2] = 0
        b[b==3] = 0
        b[b==4] = 154
        b[b==5] = 0
        b[b==6] = 0

        return cv2.merge([b, g, r])