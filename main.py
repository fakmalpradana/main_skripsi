from utils.bantu import BacaData, Dataset

img_dataset = BacaData('image').ambilRGB()
msk_dataset = BacaData('mask').ambil()
dem_dataset = BacaData('dem').ambil()

X_data, Y_data = Dataset(img_dataset, msk_dataset, dem_dataset).splitXY()

print(f'Ukuran data input (gambar dan dem) adalah {X_data.shape}')
print(f'Ukuran data label atau data mask adalah {Y_data.shape}')