import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import rasterio as rio
from datetime import datetime
from osgeo import gdal
from osgeo import osr

directorio= os.getcwd()
print("You working at directory path:" + directorio)


events = ["20200608","20200920","20210305","20210311","20210412","20210507","20211212","20220404","20231202"]
events_n_beforeafter = [[3,1],[10,1],[3,1],[3,1],[3,1],[3,1],[3,1],[3,1],[3,1]] #Allows us to change how many images are involved in the linear inversion
events_objs = []

#To do inversion I will choose the 3 images before an event and the one straight after 
# ##This was my original plan but has changed now!

#convert list of date strings to dates

for idx,date in enumerate(events):
    events_objs.append(datetime.strptime(date, '%Y%m%d'))
    

   
#convert dates on TIF files to dates objects

TIF_image_paths = glob.glob(f"{directorio}/resized_images/*.tif") #Makes sure all images are same size, this folder has come from another 
TIF_image_paths.sort()  #I didnt have this line and got errors for so long hahaha

TIF_image_dates = []
date_strings = []
for idx, TIF_image_date in enumerate(TIF_image_paths):
    date_strings.append(os.path.basename(TIF_image_date)[8:16])
    TIF_image_dates.append(datetime.strptime(date_strings[idx], '%Y%m%d'))

TIF_image_dates_array = np.array(TIF_image_dates)

events_dates = []
#Find string value of the three dates before each event and 1 after, so there should be 4 indices in each element
    
for idx,event_ob in enumerate(events_objs):
    first_greaterthan_index = np.where(TIF_image_dates_array > event_ob)[0][0]  #INDEX of the first true value in the array
    events_dates.append(date_strings[first_greaterthan_index - events_n_beforeafter[idx][0] : first_greaterthan_index + events_n_beforeafter[idx][1]])
            #EXAMPLE idx = 0: Because events_n_beforeafter[0][0] = [[3,1],.....][0][0] = 3, but the 3 actually means 2 indexs before and 0, this
                #must then have 1 added to it.
                #Also events_dates looks like: [[date1event1,date2event1,..,dateIevent1],...,[[...],....,[...]],...,[date1eventN,...,dateJeventN]]
           

events_images = []

for idx,event_ob in enumerate(events_objs):
    
    particular_event_images = []
    
    for jdx in range(events_n_beforeafter[idx][0] + events_n_beforeafter[idx][1] ):       #jdx so we dont use idx twice     #I.e. for [[3,1],...] = 3+1=4
        image_file = rio.open(f"{directorio}/TIF/conv_EQA{events_dates[idx][jdx]}.isp.mli.geo.tif").read(1)
        img = image_file 
        
        img[img == 0.] = np.nan
        img = 10 * np.log10(img)
        img[img == np.nan] = 0
        #events_images[jdx] = img  #We now have all of the images at the required dates in this variable, formatted in dB
        particular_event_images.append(img)    

    events_images.append(particular_event_images)
    
    
    
    
    #Now we need to do the linear inversion

 

#Find plot axis range

geotiff = gdal.Open(directorio+'/TIF/conv_EQA20200103.isp.mli.geo.tif') #TSX MASTER

# Boundaries of tif
ulx, xres, xskew, uly, yskew, yres = geotiff.GetGeoTransform()
lrx = ulx + (geotiff.RasterXSize * xres)
lry = uly + (geotiff.RasterYSize * yres)
extent=[ulx, lrx, lry, uly]

image_file = rio.open(f"{directorio}/TIF/conv_EQA20200103.isp.mli.geo.tif").read(1)
img = np.array(image_file)
x_pixels, y_pixels = img.shape


#Make new folder for images

LinInversionPath = os.path.join(directorio, f"LinearInversionsForSpecificEvents")
os.makedirs(LinInversionPath, exist_ok=True)
    

#This is the method Pedro reccomended for linear inversion


for idx,particular_event_images in enumerate(events_images):
    
    particular_event_nparray_images = np.array(particular_event_images)
    no_observations = len(particular_event_nparray_images)
    
    # Construct the design matrix G
    G = np.zeros((no_observations, 2))
    G[:, 1] = 1       #arange(no_observations)  # Assuming the step occurs after the first image
    G[-1,0] = 1
    
    regularization_term = 1e-12
    
    # # Solve for the model parameters m using least squares
    # m = (G.T * G)^(-1) * G.T * d
    GTG_inv = np.linalg.inv(G.T @ G + regularization_term *  np.eye(2))
    
    reshaped_images = particular_event_nparray_images.reshape(no_observations, -1)
    
    m_flat = GTG_inv @ G.T @ reshaped_images
    
    m = m_flat.reshape(2, x_pixels, y_pixels)
    
    # Extract the pre-step level and step estimation
    pre_step_level = m[1]
    step_estimation = m[0]

    backscatter_ratio_image = step_estimation / ((2 * pre_step_level) + 1) 
    
    fig, ax = plt.subplots(figsize=(15,12))
    im = ax.imshow(backscatter_ratio_image, vmin=-1,vmax=1,extent=[ulx, lrx, lry, uly],cmap='RdBu') #backscatter_ratio_image
    plt.title(f"{events[idx]}",loc = 'center',y=1.05)
    plt.axis([-78.3920,-78.2688,-2.0716,-1.9511])
    
    # cbar = plt.colorbar(im, ax = ax, orientation="horizontal", label="mm")  #replaced cax=[0.32, -0.02, 0.35, 0.02] with new ax
    # cbar.ax.tick_params(labelsize=8) 
    cax = fig.add_axes([0.32, -0.02, 0.35, 0.02])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="mm")
    
    plt.savefig(f'{LinInversionPath}/Date_{events[idx]}.png', dpi=300, format='png', transparent='true', bbox_inches='tight', pad_inches = 0)
    plt.close(fig)
    
    driver = gdal.GetDriverByName('GTiff')   
    path_output = f"{LinInversionPath}/Date_{events[idx]}.tif"
    size1,size2=backscatter_ratio_image.shape
    dataset_output=driver.Create(path_output,size2,size1,1,gdal.GDT_Float64)
    #orig_file = gdal.Open(filename, gdal.GA_ReadOnly)
    GT_output = geotiff.GetGeoTransform()
    dataset_output.SetGeoTransform(GT_output)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    dataset_output.SetProjection(srs.ExportToWkt())
    export= dataset_output.GetRasterBand(1).WriteArray(backscatter_ratio_image)
    export = None
    
