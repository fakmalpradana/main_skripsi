import cv2
import numpy as np

# Membuat class Data
class Dirname:
    def __init__(self, item, format):
        self.item = item
        self.format = format

    def list(self):
        dir = []
        item = self.item
        format = self.format
        for i in range(7):
            for j in range(9):
                dir.append(f'{item}_{i}_{j}.{format}')
        
        return dir

# membaca dataset dan memasukkannya ke dalam variabel
class BacaData:
    def __init__(self, dir):
        self.dir = dir
    
    def dirname(self, listDir):
        ROOT_DIR = f'./data/{self.dir}_patch/'
        dirfile = []

        for i in listDir:
            dirfile.append(f'{ROOT_DIR}{i}')

        return dirfile
    
    def ambil(self):
        if (self.dir == 'dem'):
            FILE_DIR = Dirname(self.dir,'tif').list()
            return [cv2.imread(x, -1) for x in self.dirname(FILE_DIR)]

        FILE_DIR = Dirname(self.dir,'png').list()
        return [cv2.imread(x) for x in self.dirname(FILE_DIR)]

    def ambilRGB(self):
        return [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in self.ambil()]

class Dataset:
    def __init__(self, img, msk, dem):
        self.img = img
        self.msk = msk
        self.dem = dem

    def splitXY(self):
        dem = np.expand_dims(np.array(self.dem), -1)
        dem = dem.round(decimals=2)
        img = np.array(self.img)

        X = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], dem])
        Y = np.array(self.msk)

        return X, Y