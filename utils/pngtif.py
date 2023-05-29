from PIL import Image
from osgeo import gdal

# convert format
img = Image.open('ugm_35.png')
img.save('ugm_35.tif')

# buka raster
ds = gdal.Open('ugm_35.tif', gdal.GA_Update)
ref = gdal.Open('ortho_clip_65.tif', gdal.GA_Update)

# get geotransform
gt1 = ds.GetGeoTransform()
gt2 = ref.GetGeoTransform()

# assign geotransform
ds.SetGeoTransform(gt2)
print(ds)

ds.FlushCache()
ds = None
# gdal_edit.py -a_srs EPSG:32749 ugm_35.tif
