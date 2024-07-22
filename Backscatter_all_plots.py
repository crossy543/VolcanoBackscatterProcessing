import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import math
import rasterio as rio
from osgeo import gdal
from skimage.transform import resize
from osgeo import osr


directorio = os.getcwd()

def last_n_avg(n, dates, directorio):
    """Takes the average plot of the previous 3 images to highlight rapid change

    Args:
        n (_type_): _description_
        dates (_type_): _description_
        directorio (_type_): _description_

    Returns:
        _type_: _description_
    """
    list_imgs = [] 
    for i in range(len(dates)):
        img = rio.open(f"{directorio}/resized_images/conv_EQA{dates[i]}.isp.mli.geo.tif").read(1)
        img[img == 0.] = np.nan
        img = 10 * np.log10(img)
        list_imgs.append(img)
        
        
    avg_img = np.nanmean(list_imgs, axis=0)
    avg_img[np.isnan(avg_img)] = 0  # Set NaN values to 0
    return avg_img


dates = []
pth = glob.glob(f"{directorio}/TIF/*.tif")
pth.sort()
count=1
for p in pth:
	date = os.path.basename(p)[8:16]
	dates.append(date)
	count += 1
number_images = count

cols=math.isqrt(number_images)+1
rows=math.isqrt(number_images)+1 



geotiff = gdal.Open(directorio+'/TIF/conv_EQA20200103.isp.mli.geo.tif') #TSX MASTER
data = geotiff.ReadAsArray()

# Boundaries of tif
ulx, xres, xskew, uly, yskew, yres = geotiff.GetGeoTransform()
lrx = ulx + (geotiff.RasterXSize * xres)
lry = uly + (geotiff.RasterYSize * yres)
extent=[ulx, lrx, lry, uly]

backscatter_all = os.path.join(directorio, "All_backscatter_images")
os.makedirs(backscatter_all, exist_ok=True)



i=0
frames_per_avg = 1

for p in range(frames_per_avg, len(dates) - frames_per_avg + 1): #+1 so you dont get the last one but just up to it
    # if i < 2:
    #     i += 1
    running_avg = last_n_avg(frames_per_avg,dates[p-frames_per_avg:p],directorio) #Only gives up to p-1'th value
    comparison = running_avg

    comparee_img = rio.open(f"{directorio}/resized_images/conv_EQA{dates[p]}.isp.mli.geo.tif").read(1)
    comparee = comparee_img
 
    comparee[comparee == 0.] = np.nan
    comparee = 10 * np.log10(comparee)
    comparee[comparee == np.nan] = 0

    sp = plt.subplot(rows,cols,i+1)
    sp.set_aspect('equal')

    r0dB = 2*(comparee-comparison)/(comparee+comparison) #Multiply by 2 from Goitom et al 2015

    fig, ax = plt.subplots(figsize=(15,12))
    im = ax.imshow(r0dB, vmin=-1,vmax=1,extent=[ulx, lrx, lry, uly],cmap='RdBu') #backscatter_ratio_image
    
    plt.title(f"{dates[p]}",loc = 'center',y=1.05)
    plt.axis([-78.3920,-78.2688,-2.0716,-1.9511])
    
    # cbar = plt.colorbar(im, ax = ax, orientation="horizontal", label="mm")  #replaced cax=[0.32, -0.02, 0.35, 0.02] with new ax
    # cbar.ax.tick_params(labelsize=8) 
    cax = fig.add_axes([0.32, -0.02, 0.35, 0.02])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="mm")
    
    plt.savefig(f'{backscatter_all}/{dates[p]}_backscatter_change.png', dpi=300, format='png', transparent='true', bbox_inches='tight', pad_inches = 0)
    plt.close(fig)
    
    
    driver = gdal.GetDriverByName('GTiff')   
    path_output = f"{backscatter_all}/Date_{dates[i]}.tif"
    
    # size1,size2=r0dB.shape
    # dataset_output=driver.Create(path_output,size2,size1,1,gdal.GDT_Float64)
    # #orig_file = gdal.Open(filename, gdal.GA_ReadOnly)
    # GT_output = geotiff.GetGeoTransform()
    # dataset_output.SetGeoTransform(GT_output)
    # srs = osr.SpatialReference()
    # srs.SetWellKnownGeogCS('WGS84')
    # dataset_output.SetProjection(srs.ExportToWkt())
    # export= dataset_output.GetRasterBand(1).WriteArray(r0dB)
    # export = None
    
    
    with gdal.GetDriverByName('GTiff').Create(path_output, r0dB.shape[1], r0dB.shape[0], 1, gdal.GDT_Float64) as dataset_output:
        dataset_output.SetGeoTransform(geotiff.GetGeoTransform())
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS('WGS84')
        dataset_output.SetProjection(srs.ExportToWkt())
        dataset_output.GetRasterBand(1).WriteArray(r0dB)
    
    
    
    i += 1

