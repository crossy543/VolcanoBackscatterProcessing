import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import math
import rasterio as rio
from osgeo import gdal
from skimage.transform import resize


directorio= os.getcwd()
print("You working at directory path:" + directorio)

#Numerate the files
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
rows=math.isqrt(number_images)+1 #integer square root of no. images to help with later plotting


#EXTENSION
geotiff = gdal.Open(directorio+'/TIF/conv_EQA20200103.isp.mli.geo.tif') #TSX MASTER
data = geotiff.ReadAsArray()

# Boundaries of tif
ulx, xres, xskew, uly, yskew, yres = geotiff.GetGeoTransform()
lrx = ulx + (geotiff.RasterXSize * xres)
lry = uly + (geotiff.RasterYSize * yres)
extent=[ulx, lrx, lry, uly]


pth_TIF = directorio+'/TIF/'
minvamat=[]
maxvamat=[]
i=0
for p in pth:
	datesname = dates[i]
    #Open every tif
	image_file = rio.open(f"{directorio}/TIF/conv_EQA{datesname}.isp.mli.geo.tif").read(1)
	img = image_file
	pv_figure = plt.figure(figsize=(18, 15))
	img[img == 0.] = np.nan # Filters out pixels with value 0 for log
	#change to log for plotting
	r0dB = 10 * np.log10(img)
	minvamat.append(np.nanmin(r0dB))
	maxvamat.append(np.nanmax(r0dB))
	i=i+1
	plt.close()
 
maxplot=np.nanmax(maxvamat)
minplot=np.nanmin(minvamat)
meanplot=np.nanmean(minvamat)


## PLOT ALL TIFS #####################################################################################################################################
fig, ax=plt.subplots(figsize=(15,12))
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0.1)
plt.margins(0,0)
i=0

for p in pth:
    datesname = dates[i]
    image_file = rio.open(f"{directorio}/TIF/conv_EQA{datesname}.isp.mli.geo.tif").read(1)
    img = image_file
    
    img_wrongsize = []
    array = np.array(img)                       # USED TO SHOW SHAPE OF IMAGES WAS ALL (676,980)
    
    if array.shape != (676,980):
        img_wrongsize.append(i)
    
    subplot = plt.subplot(rows, cols, i + 1)
    subplot.set_aspect('equal')  # Set ratio between row & col to 1

    img[img == 0] = np.nan
    r0dB = 10 * np.log10(img)
    
    plt.axis([-78.3920, -78.2688, -2.0716, -1.9511])
    im = plt.imshow(r0dB, cmap='gray', vmin=minplot, vmax=maxplot, extent=[ulx, lrx, lry, uly])
    plt.title(f"{datesname}")
    plt.axis('off')
    i += 1
    
fig.tight_layout()
cax = fig.add_axes([0.32, -0.02, 0.35, 0.02])
fig.colorbar(im, cax=cax, orientation="horizontal", label="mm")
plt.savefig(f'{directorio}/TSX_similarscale.png', dpi=300, format='png', transparent='true', bbox_inches='tight', pad_inches = 0) #DPI ??






#########################################################################Functions! ######################################################################################################
def resize_and_save_images(dates, directorio, img_index_wrongsize, target_shape=(676, 980)):
    """Takes all of the images in directorio, with path f"TIF/conv_EQA{date}.isp.mli.geo.tif", and makes a new folder in directorio called resized_images
    Args:
        dates (list): list of all dates from file names
        directorio (str): directory with the TIF folder and this script
        img_index_wrongsize (list): indexs of date in dates with incorrect size
        target_shape (tuple, optional): Target shape. Defaults to (676, 980).
    """
    # Create a new directory for resized images
    resized_dir = os.path.join(directorio, "resized_images")
    os.makedirs(resized_dir, exist_ok=True)

    for idx, date in enumerate(dates):
        img_path = os.path.join(directorio, f"TIF/conv_EQA{date}.isp.mli.geo.tif")
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
        resized_img_path = os.path.join(resized_dir, f"conv_EQA{date}.isp.mli.geo.tif")
        with rio.open(resized_img_path, 'w', **profile) as dst:
            dst.write(img, 1)

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


#################################################### PLOT Difference between each date to the average of the n before ####################################################
resize_and_save_images(dates, directorio, img_wrongsize)

fig, ax=plt.subplots(figsize=(15,12))
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0.1)
plt.margins(0,0)
          

i=0
frames_per_avg = 4

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

    
    plt.axis([-78.3920,-78.2688,-2.0716,-1.9511])
    im = plt.imshow(r0dB, vmin=-1,vmax=1,extent=[ulx, lrx, lry, uly],cmap='RdBu')
    plt.title(f"{dates[p]}",loc = 'center',y=0)
    plt.axis('off')
    #plt.text(0.5, -0.1, "{}".format(str(dates[p])), ha='center', transform=subplot.transAxes)
    
    if dates[p] == '20200903':
        print('jytfytf uyfytdyrsrstrsredt')
        fig2, ax2 = plt.subplots(figsize=(15,12))
        im = ax2.imshow(r0dB, vmin=-1,vmax=1,extent=[ulx, lrx, lry, uly],cmap='RdBu') #backscatter_ratio_image
        plt.title("20200903",loc = 'center',y=1.05)
        plt.axis([-78.3920,-78.2688,-2.0716,-1.9511])
        
        cax2 = fig2.add_axes([0.32, -0.02, 0.35, 0.02])
        fig2.colorbar(im, cax=cax2, orientation="horizontal", label="mm")
        
        plt.savefig(f'{directorio}/20200903SingleRGB.jpg', dpi=300, format='jpg', transparent='true', bbox_inches='tight', pad_inches = 0)
        plt.close(fig2)
        
    i += 1

cax = fig.add_axes([0.32, -0.02, 0.35, 0.02])
fig.colorbar(im, cax=cax, orientation="horizontal", label="mm")
plt.savefig(f'{directorio}/difference_{frames_per_avg}_avg.jpg', dpi=300, format='jpg', transparent='true', bbox_inches='tight', pad_inches = 0)



 
 



