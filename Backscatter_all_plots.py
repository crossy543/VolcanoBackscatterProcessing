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

backscatter_all_speckle = os.path.join(speckle_path, "All_backscatter_images_speckle3x3")
os.makedirs(backscatter_all_speckle, exist_ok=True)



def resize_and_save_images(dates , img_index_wrongsize, directory_func = directorio,raw_images_path = raw_images_path, target_shape=(676, 980),folder_tosave_name = "resized_images"):
    """Resaves all files of form to same size in folder spoecified

    Args:
        dates (list): dates of images in fomr yyyymmdd
        img_index_wrongsize (list): indiees of images wiht size different to the first image
        directory_func (str, optional): base directory of which the raw imgs are saved as  SpeckleFilter_TIFs and where the new imgs are to be saved. Defaults to directorio.
        raw_images_path (str, optional): path where all the raw images are saved to. Defaults to raw_images_path.
        target_shape (tuple, optional): target shape of images, in pixels. Defaults to (676, 980).
        folder_tosave_name (str, optional): name of folder to create to save new images in. Defaults to "resized_images".
    """

    # Create a new directory for resized images
    resized_dir = os.path.join(directory_func, folder_tosave_name)
    os.makedirs(resized_dir, exist_ok=True)

    for idx, date in enumerate(dates):
        img_path = os.path.join(raw_images_path, f"conv_EQA{date}.isp.nmli.geo.tif")
        
        with rio.open(img_path) as src:
            img = src.read(1)
            profile = src.profile  # Get the profile for saving the image later

        # Resize image if it's in the list of wrong-sized indices
        if idx in img_index_wrongsize:  
            img = resize(img, target_shape, preserve_range=True, anti_aliasing=True)

        # Update profile to new shape
        profile.update({
            'height': target_shape[0],
            'width': target_shape[1]
        })

        # Save the image to the new directory
        resized_img_path = os.path.join(resized_dir, f"conv_EQA{date}.isp.nmli.geo.tif")
        with rio.open(resized_img_path, 'w', **profile) as dst:
            dst.write(img, 1)
            
            
def last_n_avg(n, dates, directory_func = directorio,resized_folder_name = "resized_images"):
    """Takes the average plot of the previous 3 images to highlight rapid 

    Args:
        n (_type_): _description_
        dates (_type_): _description_
        directorio (_type_): _description_

    Returns:
        _type_: _description_
    """
    list_imgs = [] 
    for i in range(len(dates)):
        img = rio.open(f"{directory_func}/{resized_folder_name}/conv_EQA{dates[i]}.isp.nmli.geo.tif").read(1)
        img[img <= 0.] = np.nan
        img = 10 * np.log10(img)
        list_imgs.append(img)

    avg_img = np.nanmean(list_imgs, axis=0)
    
    return avg_img


dates = []
pth = glob.glob(f"{raw_images_path}/conv_EQA*.isp.nmli.geo.tif")
pth.sort()

count=1
for p in pth:
	date = os.path.basename(p)[8:16]
	dates.append(date)
	count += 1
number_images = count

cols=math.isqrt(number_images)+1
rows=math.isqrt(number_images)+1 

geotiff = gdal.Open(f'{raw_images_path}/conv_EQA20200103.isp.nmli.geo.tif') #TSX MASTER
data = geotiff.ReadAsArray()

# Boundaries of tif
ulx, xres, xskew, uly, yskew, yres = geotiff.GetGeoTransform()
lrx = ulx + (geotiff.RasterXSize * xres)
lry = uly + (geotiff.RasterYSize * yres)
extent=[ulx, lrx, lry, uly]



i=0
img_wrongsize = []
img_shapes = []
for p in pth:
    datesname = dates[i]
    image_file = rio.open(f"{raw_images_path}/conv_EQA{datesname}.isp.nmli.geo.tif").read(1)
    img = image_file
    
    array = np.array(img)  
    
    if i == 0:
        shape_firstimg = array.shape
     
    if array.shape != shape_firstimg:   #(676,980): is nbormally this shape
        img_wrongsize.append(i)
        
    i += 1
    


frames_per_avg = 1
new_folder_name = "resized_images_nobaseline_speckle3x3"
resize_and_save_images(dates, img_wrongsize, directory_func = speckle_path, raw_images_path = raw_images_path, target_shape=shape_firstimg, folder_tosave_name = new_folder_name)

for p in range(frames_per_avg, len(dates) - frames_per_avg + 1): #+1 so you dont get the last one but just up to it

    running_avg = last_n_avg(frames_per_avg, dates[p-frames_per_avg:p], directory_func = speckle_path, resized_folder_name=new_folder_name) #Only gives up to p-1'th value
    comparison = running_avg

    comparee_img = rio.open(f"{speckle_path}/{new_folder_name}/conv_EQA{dates[p]}.isp.nmli.geo.tif").read(1)
    comparee = comparee_img

    comparee[comparee <= 0.] = np.nan
    comparee = 10 * np.log10(comparee)

    r0dB = 2*(comparee-comparison)/(comparee+comparison) #Multiply by 2 from Goitom et al 2015

    r0dB[np.where(((r0dB>=-0.1) & (r0dB<=0.1)))] = np.nan

    fig, ax = plt.subplots(figsize=(15,12))
    im = ax.imshow(r0dB, vmin=-1,vmax=1,extent=[ulx, lrx, lry, uly],cmap='RdBu') #backscatter_ratio_image
    
    plt.title(f"{dates[p]}",loc = 'center',y=1.05)
    plt.axis([-78.3920,-78.2688,-2.0716,-1.9511])
    
    cax = fig.add_axes([0.32, -0.02, 0.35, 0.02])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="mm")
    
    plt.savefig(f'{backscatter_all_speckle}/{dates[p]}_backscatter_change.png', dpi=300, format='png', transparent='true', bbox_inches='tight', pad_inches = 0)
    plt.close(fig)
    
    driver = gdal.GetDriverByName('GTiff')   
    path_output = f"{backscatter_all_speckle}/{dates[p]}_backscatter_change.tif"
    
    with gdal.GetDriverByName('GTiff').Create(path_output, r0dB.shape[1], r0dB.shape[0], 1, gdal.GDT_Float64) as dataset_output:
        dataset_output.SetGeoTransform(geotiff.GetGeoTransform())
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS('WGS84')
        dataset_output.SetProjection(srs.ExportToWkt())
        dataset_output.GetRasterBand(1).WriteArray(r0dB)


