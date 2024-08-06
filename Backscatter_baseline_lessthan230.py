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
speckle_path = os.path.join(directorio + '/Speckle_Filter_Applied3x3')
raw_images_path = os.path.join(directorio + '/Speckle_Filter_Applied3x3/SpeckleFilter3x3_TIFs')

backscatter_lowbaseline = os.path.join(speckle_path, "Low_baseline_backscatter_images_speckle3x3")
os.makedirs(backscatter_lowbaseline, exist_ok=True)


def last_n_avg(n, dates, directory_func = directorio,resized_folder_name = "resized_images"):
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
        #img = rio.open(f"{directorio}/resized_images/conv_EQA{dates[i]}.isp.mli.geo.tif").read(1)
        
        img = rio.open(f"{directory_func}/{resized_folder_name}/conv_EQA{dates[i]}.isp.nmli.geo.tif").read(1)
        
        img[img <= 0.] = np.nan
        img = 10 * np.log10(img)
        list_imgs.append(img)
        
        
    avg_img = np.nanmean(list_imgs, axis=0)

    return avg_img


dates = []
pth = glob.glob(f"{raw_images_path}/conv_EQA*.isp.nmli.geo.tif")
pth.sort()

pth_array = np.array(pth)

count=1
for p in pth:
	date = os.path.basename(p)[8:16]
	dates.append(date)
	count += 1
number_images = count

cols=math.isqrt(number_images)+1
rows=math.isqrt(number_images)+1 

baseline_bperp_strings = np.genfromtxt(f'{directorio}/baseline_data.csv',delimiter=',', dtype=None, encoding=None)

baseline_bperp_dates = baseline_bperp_strings['f0'].astype(str)
baseline_bperp_values = baseline_bperp_strings['f1'].astype(float)

low_baseline_dates = baseline_bperp_dates[abs(baseline_bperp_values) <= 230]    #value looks best from image 

bool_good_paths_from_baseline = np.isin(dates, low_baseline_dates)   

good_paths = pth_array[bool_good_paths_from_baseline]

geotiff = gdal.Open(raw_images_path+'/conv_EQA20200103.isp.nmli.geo.tif') #TSX MASTER
data = geotiff.ReadAsArray()

# Boundaries of tif
ulx, xres, xskew, uly, yskew, yres = geotiff.GetGeoTransform()
lrx = ulx + (geotiff.RasterXSize * xres)
lry = uly + (geotiff.RasterYSize * yres)
extent=[ulx, lrx, lry, uly]


frames_per_avg = 1

for p in range(frames_per_avg, len(good_paths) - frames_per_avg + 1): #+1 so you dont get the last one but just up to it
    running_avg = last_n_avg(frames_per_avg, dates[p-frames_per_avg:p], directory_func = speckle_path, resized_folder_name='resized_images_nobaseline_speckle3x3') #Only gives up to p-1'th value
    comparison = running_avg

    comparee_img = rio.open(f"{speckle_path}/resized_images_nobaseline_speckle3x3/conv_EQA{low_baseline_dates[p]}.isp.nmli.geo.tif").read(1)
    comparee = comparee_img

    comparee[comparee <= 0.] = np.nan
    comparee = 10 * np.log10(comparee)

    r0dB = 2*(comparee-comparison)/(comparee+comparison) #Multiply by 2 from Goitom et al 2015

    r0dB[np.where(((r0dB>=-0.1) & (r0dB<=0.1)))] = np.nan

    fig, ax = plt.subplots(figsize=(15,12))
    im = ax.imshow(r0dB, vmin=-1,vmax=1,extent=[ulx, lrx, lry, uly],cmap='RdBu') #backscatter_ratio_image
    
    plt.title(f"{low_baseline_dates[p]}",loc = 'center',y=1.05)
    plt.axis([-78.3920,-78.2688,-2.0716,-1.9511])
    
    cax = fig.add_axes([0.32, -0.02, 0.35, 0.02])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="mm")
    
    plt.savefig(f'{backscatter_lowbaseline}/{low_baseline_dates[p]}_backscatter_change.png', dpi=300, format='png', transparent='true', bbox_inches='tight', pad_inches = 0)
    plt.close(fig)
    
    
    driver = gdal.GetDriverByName('GTiff')   
    path_output = f"{backscatter_lowbaseline}/{low_baseline_dates[p]}_backscatter_change.tif"
        
    with gdal.GetDriverByName('GTiff').Create(path_output, r0dB.shape[1], r0dB.shape[0], 1, gdal.GDT_Float64) as dataset_output:
        dataset_output.SetGeoTransform(geotiff.GetGeoTransform())
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS('WGS84')
        dataset_output.SetProjection(srs.ExportToWkt())
        dataset_output.GetRasterBand(1).WriteArray(r0dB)
    


