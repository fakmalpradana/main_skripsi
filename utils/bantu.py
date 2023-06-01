import cv2
import numpy as np
import patchify as p

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
    def __init__(self, dir:str):
        img = cv2.imread(dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        
        self.dir = dir
        self.size_ori = img.shape
        self.img = img
    
    def loadData(self):
        return self.img
    # def dirname(self, listDir):
    #     ROOT_DIR = f'./data/{self.dir}_patch/'
    #     dirfile = []

    #     for i in listDir:
    #         dirfile.append(f'{ROOT_DIR}{i}')

    #     return dirfile
    
    # def ambil(self):
    #     if (self.dir == 'dem'):
    #         FILE_DIR = Dirname(self.dir,'tif').list()
    #         return [cv2.imread(x, -1) for x in self.dirname(FILE_DIR)]

    #     FILE_DIR = Dirname(self.dir,'png').list()
    #     return [cv2.imread(x) for x in self.dirname(FILE_DIR)]

    # def ambilRGB(self):
    #     return [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in self.ambil()]
    
    def patchData(self, patch_dim:int, step_size:float):
        img = self.img

        # pembulatan channel img
        ch1 = np.ceil(img.shape[0]/patch_dim).astype(int)
        ch2 = np.ceil(img.shape[1]/patch_dim).astype(int)

        # menambahkan kolom dan baris kosong pd gambar
        arr0 = np.zeros((ch1*patch_dim, ch2*patch_dim, 3))
        arr0[:img.shape[0], :img.shape[1]] += img
        arr = arr0.astype(np.uint8)

        # patch image
        patch_shape = (patch_dim, patch_dim, 3)
        patches = p.patchify(arr, patch_shape, step=int(patch_dim*step_size))

        img_patches = []
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                img_patches.append(patches[i,j,0,:,:,:])

        return img_patches
    
    def sizeDataOri(self):
        return self.size_ori

class Dataset:
    def __init__(self, img, msk, hsd, sgm):
        self.img = img
        self.msk = msk
        self.hsd = hsd
        self.sgm = sgm

    def splitXY(self):
        hsd = np.array(self.hsd)
        img = np.array(self.img)
        sgm = np.array(self.sgm)

        X = np.concatenate((hsd, img, sgm), axis=-1)
        Y = np.array(self.msk)

        return X, Y