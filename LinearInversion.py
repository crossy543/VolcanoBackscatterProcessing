import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import rasterio as rio
from datetime import datetime
from osgeo import gdal
from osgeo import osr
import warnings

directorio= os.getcwd()
speckle_string = "speckle3x3"
speckle_path = os.path.join(directorio + '/Speckle_Filter_Applied3x3')

all_backscatter_images = os.path.join(speckle_path + f'/All_backscatter_images_{speckle_string}')
lowbaseline_backscatter_images = os.path.join(speckle_path + f'/Low_baseline_backscatter_images_{speckle_string}')
resized_images = os.path.join(speckle_path + f'/resized_images_nobaseline_{speckle_string}')

chosen_images = all_backscatter_images

LinInversionPath = os.path.join(f"{chosen_images}_LinearInversionsForSpecificEvents")  #Where images are saved to!
os.makedirs(LinInversionPath, exist_ok=True)

events = ["20200608","20200920","20210305","20210311","20210412","20210507","20211212","20220404","20231202"]
events_objs = []

#To do inversion I will choose the 3 images before an event and the one straight after 
# ##This was my original plan but has changed now!

#convert list of date strings to dates

for idx,date in enumerate(events):
    events_objs.append(datetime.strptime(date, '%Y%m%d'))
    
#convert dates on TIF files to dates objects

TIF_image_paths = glob.glob(f"{chosen_images}/*.tif") 
TIF_image_paths.sort()  #I didnt have this line and got errors for so long hahaha

TIF_image_dates = []
date_strings = []

for idx, TIF_image_date in enumerate(TIF_image_paths):
    date_strings.append(os.path.basename(TIF_image_date)[0:8])
    TIF_image_dates.append(datetime.strptime(date_strings[idx], '%Y%m%d'))

TIF_image_dates_array = np.array(TIF_image_dates)

#################Find best prev_event_dates and post_event_dates ###################

indices_after_events = []



for event_ob in events_objs:
    indices_after_events.append(np.where(TIF_image_dates_array > event_ob)[0][0])     #INDEX of the first true value in the array

differences = np.append(np.diff(indices_after_events),len(TIF_image_dates)-indices_after_events[-1])

steps_prev_default = 3
steps_post_default = 1

events_n_beforeafter = []
events_dates = []

for idx, event_index in enumerate(indices_after_events):
    if idx == 0:
        bounds_lower = int(event_index) if event_index < 3 else steps_prev_default
        
        if differences[idx] <= 2: #I.e. if there is less than 2 images taken between 2 events the inversion may still include tefra from the previous eruption event!
            warnings.warn(f"The events on {events[idx]}, {events[idx+1]} do not have enough data between them to give reliable Linear Inversion!")
    
        
    bounds_lower = int(differences[idx-1]) if differences[idx-1] < 3 else steps_prev_default
    
    if differences[idx] <= 2: 
            warnings.warn(f"The events on {events[idx]}, {events[idx+1]} do not have enough data between them to give reliable Linear Inversion!")
    
    bounds_upper = 1
    events_n_beforeafter.append([bounds_lower,bounds_upper])


for idx, event_ob in enumerate(events_objs):
    index = indices_after_events[idx]
    bounds_lower = events_n_beforeafter[idx][0]
    bounds_upper = events_n_beforeafter[idx][1]
    
    lower_bound_index = index - bounds_lower
    upper_bound_index = index + bounds_upper
    
    events_dates.append(date_strings[lower_bound_index:upper_bound_index + 1])

    

#EXAMPLE idx = 0: Because events_n_beforeafter[0][0] = [[3,1],.....][0][0] = 3, but the 3 actually means 2 indexs before and 0, this
#must then have 1 added to it.
#Also events_dates looks like: [[date1event1,date2event1,..,dateIevent1],...,[[...],....,[...]],...,[date1eventN,...,dateJeventN]]

events_images = []

for idx,event_ob in enumerate(events_objs):
    
    particular_event_images = []
    
    for jdx in range(events_n_beforeafter[idx][0] + events_n_beforeafter[idx][1]):       #jdx so we dont use idx twice     #I.e. for [[3,1],...] = 3+1=4
        image_file = rio.open(f"{all_backscatter_images}/{events_dates[idx][jdx]}_backscatter_change.tif").read(1)
        img = image_file 
          
        particular_event_images.append(img)    

    events_images.append(particular_event_images)

######################Now we need to do the linear inversion#####################


#Find plot axis range

geotiff = gdal.Open(all_backscatter_images+'/20200205_backscatter_change.tif') #TSX MASTER

# Boundaries of tif
ulx, xres, xskew, uly, yskew, yres = geotiff.GetGeoTransform()
lrx = ulx + (geotiff.RasterXSize * xres)
lry = uly + (geotiff.RasterYSize * yres)
extent=[ulx, lrx, lry, uly]


image_file = rio.open(f"{all_backscatter_images}/20200205_backscatter_change.tif").read(1)
img = np.array(image_file)
x_pixels, y_pixels = img.shape



    

#This is the method Pedro reccomended for linear inversion

for idx,particular_event_images in enumerate(events_images):
    
    particular_event_nparray_images = np.array(particular_event_images)
    no_observations = len(particular_event_nparray_images)

    # Construct the design matrix G
    G = np.zeros((no_observations, 2))
    G[:, 1] = 1     ########################### Assuming the step occurs after the first image ##########################
    G[-1,0] = 1
    
    regularization_term = 1e-10
    
    ########################## Solve for the model parameters m using least squares##########################
    GTG_inv = np.linalg.inv(G.T @ G + regularization_term *  np.eye(2))
    
    reshaped_images = particular_event_nparray_images.reshape(no_observations, -1)
    
    m_flat = GTG_inv @ G.T @ reshaped_images
    
    m = m_flat.reshape(2, x_pixels, y_pixels)
    
    # Extract the pre-step level and step estimation
    pre_step_level = m[1]
    step_estimation = m[0]
    
    backscatter_ratio_image = 2 * step_estimation / ((2 * pre_step_level) + step_estimation) 
    
    fig, ax = plt.subplots(figsize=(15,12))
    im = ax.imshow(backscatter_ratio_image, vmin=-1,vmax=1,extent=[ulx, lrx, lry, uly],cmap='RdBu') #backscatter_ratio_image
    plt.title(f"{events[idx]}",loc = 'center',y=1.05)
    plt.axis([-78.3920,-78.2688,-2.0716,-1.9511])
    
    cax = fig.add_axes([0.32, -0.02, 0.35, 0.02])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="mm")
    
    plt.savefig(f'{LinInversionPath}/Date_{events[idx]}.png', dpi=300, format='png', transparent='true', bbox_inches='tight', pad_inches = 0)
    plt.close(fig)
    
    driver = gdal.GetDriverByName('GTiff')   
    path_output = f"{LinInversionPath}/Date_{events[idx]}.tif"
    size1,size2=backscatter_ratio_image.shape
    dataset_output=driver.Create(path_output,size2,size1,1,gdal.GDT_Float64)

    GT_output = geotiff.GetGeoTransform()
    dataset_output.SetGeoTransform(GT_output)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    dataset_output.SetProjection(srs.ExportToWkt())
    export= dataset_output.GetRasterBand(1).WriteArray(backscatter_ratio_image)
    export = None
    
