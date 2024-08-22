import numpy as np
import glob 
import os 
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates 
import rasterio as rio
from osgeo import gdal
from skimage.transform import resize
from osgeo import osr
import warnings
from sklearn import preprocessing
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

directorio = os.getcwd()




#################Variables to define before executing##################
speckle_extent = '3x3'          #Speckle filter which is applied to raw data before loading here
backscatter_analysis_folder_name = f"Backscatter_Analysis_Results_speckle_{speckle_extent}"          #Suggested folder name for all results to be saved into
raw_data_path = 'BackscatterAnalysisRawData' #IMPORTANT, you must make this folder and add the baseline values and raw TIF images, 
#                                            and change the names so the code will read them. You can add all the data from multiple speckle filters to this folder, 
#                                            and then at the start of this code just specify whether you want to use 2x2, 3x3 etc...

raw_TIFs_folder_name = "SpeckleFilter3x3_RawData"
raw_baseline_folder_name = "Baselines_RawData"
raw_DEM_features_folder_name = "DEM_features"

raw_tif_path_format = ["conv_EQA",".isp.nmli.geo.tif"]

events = ["20200608","20200920","20210305","20210311","20210412","20210507","20211212","20220404","20231202"]

DEM_filename = "DEM.tif"
aspect_filename = "viz.hh_aspect.tif"
roughness_filename = "viz.hh_roughness.tif"
slope_filename = "viz.hh_slope.tif"


#Clustering weightings

coordinate_weighting = 3
DEM_weighting = 1.5
slope_weighting = 1.5
roughness_weighting = 1.5
aspect_weighting = 0.5



number_random_seeds_for_clustering = 4
clusters_to_plot_separate = [4,8,12]

cluster_min = 4
cluster_max = 15

#Clusters to plot spearate must be between cluster min and cluster max!!




######dont worry from here

start_of_date = len(raw_tif_path_format[0])  #How many characters through the filename is the date

seeds = []
for i in range(number_random_seeds_for_clustering):
    seeds.append(np.random.randint(low = 2,high = 4294967290))
    

###############Create paths and directories#############

raw_images_path = os.path.join(directorio, raw_data_path, raw_TIFs_folder_name)
baselinedata_folder_path = os.path.join(directorio, raw_data_path, raw_baseline_folder_name)     
DEM_folder_path =  os.path.join(directorio, raw_data_path, raw_DEM_features_folder_name) 

backscatter_analysis_folder = os.path.join(directorio, backscatter_analysis_folder_name)
os.makedirs(backscatter_analysis_folder,exist_ok=True)

resized_backscatter_images = os.path.join(backscatter_analysis_folder, 'Resized_Raw_Images')
os.makedirs(resized_backscatter_images,exist_ok=True)

baseline_plot_path = os.path.join(backscatter_analysis_folder,'baseline_median_plot.png')
baseline_data_path = os.path.join(backscatter_analysis_folder,'baseline_median_data.csv')

backscatter_lowbaseline = os.path.join(backscatter_analysis_folder, f"Low_baseline_backscatter_images_speckle{speckle_extent}")
os.makedirs(backscatter_lowbaseline, exist_ok=True)

example_path_for_geotransform = os.path.join(resized_backscatter_images,'conv_EQA20200103.isp.nmli.geo.tif')

ratio_change = os.path.join(backscatter_analysis_folder, 'ratio_change')
os.makedirs(ratio_change,exist_ok = True)

absolute_change = os.path.join(backscatter_analysis_folder, 'absolute_change')
os.makedirs(absolute_change,exist_ok = True)

ratio_change_inpercentile = os.path.join(backscatter_analysis_folder, 'ratio_change_inpercentile')
os.makedirs(ratio_change_inpercentile,exist_ok = True)

# absolute_change_inpercentile = os.path.join(backscatter_analysis_folder, 'absolute_change_inpercentile')
# os.makedirs(absolute_change_inpercentile,exist_ok = True)

LinInversionPath = os.path.join(backscatter_analysis_folder, "LinearInversionsForSpecificEvents") 
os.makedirs(LinInversionPath, exist_ok=True)

LinInversionPathReduced = os.path.join(backscatter_analysis_folder, "LinearInversionsReducedNoiseForSpecificEvents") 
os.makedirs(LinInversionPathReduced, exist_ok=True)

where_to_save_clusterdata = os.path.join(backscatter_analysis_folder,
                                         f"bigevents_clustering_data_{DEM_weighting}_{slope_weighting}_{roughness_weighting}_{aspect_weighting}_{coordinate_weighting}.csv")


cluster_paths = []  #Folders for each set of clustering, This allows three individual random initialisations of clustering results to be generated
cluster_plot_separate = []  #file to save clusters separate, the neames of the files in each set of results which will conatin the separately plotted results

for seed in seeds:

    cluster_path = os.path.join(backscatter_analysis_folder,f"Clustering_Figures_randomconfig{seed}")
    os.makedirs(cluster_path,exist_ok = True)
    cluster_paths.append(cluster_path)
    
    plot_separate_paths = []
    
    for cluster in clusters_to_plot_separate:
        
        path_to_save_infile_clusters_separate = f"Cluster_{cluster}_plotted_separately"

        file_to_save_clusters_separate = os.path.join(cluster_path, path_to_save_infile_clusters_separate)
        plot_separate_paths.append(file_to_save_clusters_separate)
        
        os.makedirs(file_to_save_clusters_separate,exist_ok=True)
    
    cluster_plot_separate.append(plot_separate_paths)
    





##############Variables to specify after exploring data################

percentile_selection = 3 #Percentile of data to be present in linear inversion reduced image for clustering
#                          for example, if percentile_selection = 15, then data below lower 15th and above upper 85th 
#                          is present in linear inversion reduced image

average_image_shape=(676, 980)
max_baseline = 230
#Steps for linear inversion, change steps_prev_default for experimenting but not steps_post_default!
steps_prev_default = 2
steps_post_default = 1

#file_to_remove = os.path.join(directorio, "BASELINES", "20200103_20211115.base.perp") #This file has no data in!! I will include it in the last dataset but just good to know

################Useful functions for the code################


def resize_and_save_images(dates, img_index_wrongsize,folder_tosave_name = resized_backscatter_images,
                           raw_images_path = raw_images_path, target_shape=average_image_shape,path_format = raw_tif_path_format):
    
    """Resaves all files of form to same size in folder specified

    Args:
        dates (list): dates of images in fomr yyyymmdd
        img_index_wrongsize (list): indiees of images wiht size different to the first image
        directory_func (str, optional): base directory of which the raw imgs are saved as  SpeckleFilter_TIFs and where the new imgs are to be saved. Defaults to directorio.
        raw_images_path (str, optional): path where all the raw images are saved to. Defaults to raw_images_path.
        target_shape (tuple, optional): target shape of images, in pixels. Defaults to (676, 980).
        folder_tosave_name (str, optional): name of folder to create to save new images in. Defaults to "resized_images".
    """
    for idx, date in enumerate(dates):
        img_path = os.path.join(raw_images_path, path_format[0]+date+path_format[1])
        
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
        resized_img_path = os.path.join(folder_tosave_name, f"conv_EQA{date}.isp.nmli.geo.tif")
        with rio.open(resized_img_path, 'w', **profile) as dst:
            dst.write(img, 1)
                        
def last_n_avg(low_baseline_paths):
    """Takes the average plot of the previous 3 images to highlight rapid change
    Args:
        dates (list): list of dates to average over
        resized_data_path (str): folder path that resized images are saved in
        path_format (list): the format of the dates broken down into left of date and right of date, i,e. ["conv_EQA",".isp.nmli.geo.tif"]
    Returns:
        avg_img: average image from specified dates
    """
    list_imgs = [] 
    for low_baseline_path in low_baseline_paths:
        img = rio.open(low_baseline_path).read(1)
        
        img[img <= 0.] = np.NaN
        img = 10 * np.log10(img)
        list_imgs.append(img)
        
    avg_img = np.nanmean(list_imgs, axis=0)

    return avg_img

def plot_change(image, path, date,range,geotiff,extent=[-78.3920,-78.2688,-2.0716,-1.9511]):
    
    """Saves png and TIF images given the image as an array. Saves having all the code many times throughout the script!

    Raises:
        RuntimeError: Raises if driver cannot create path output to save TIF to
    """
    
    
    fig, ax = plt.subplots(figsize=(15,12))
    im = ax.imshow(image, vmin=range[0],vmax=range[1],extent=extent,cmap='RdBu') 
    plt.title(f"{date}",loc = 'center',y=1.05)
    plt.axis(extent)
    fig.colorbar(im, ax=ax, orientation="horizontal", label="mm",fraction=0.046, pad=0.04)
    plt.savefig(path+'.png', dpi=300, format='png', transparent='true', bbox_inches='tight')
    plt.close(fig)
    
    driver = gdal.GetDriverByName('GTiff')   
    path_output = path + '.tif'
    dataset_output = driver.Create(path_output, image.shape[1], image.shape[0], 1, gdal.GDT_Float64)
    if dataset_output is None:
        raise RuntimeError(f"Failed to create {path_output}")
    dataset_output.SetGeoTransform(geotiff.GetGeoTransform()) # Set GeoTransform and Projection
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    dataset_output.SetProjection(srs.ExportToWkt())
    dataset_output.GetRasterBand(1).WriteArray(image)   # Write the array to the raster band
    dataset_output.FlushCache() # Flush and close the dataset
    dataset_output = None 
    
    
###############Find bperp values and save them as csv file###########

# Define the file path
file_paths = glob.glob(f"{baselinedata_folder_path}/*.perp")
file_paths.sort()

baseline_bperp_values = []
date_objects_baseline = []
date_strings_baseline = []

for idx, file_path in enumerate(file_paths):      #Reads the bperp column and then finds median value. Returns this alongside date of acquisition
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Find the start and end of the data
        start_line = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('line'):
                start_line = i + 2  # Skip header lines
                break
        
        # Determine the total lines in the file and number of metadata lines at the end
        total_lines = len(lines)
        end_line = total_lines
        for i in range(total_lines - 1, 0, -1):
            if lines[i].strip() == '' or not lines[i][0].isdigit():
                end_line = i
            else:
                break
            
    #This code will detect metadat but in this case I just use the skip footer and header assuming that theya are the saem in each file 
            
            
    data = np.genfromtxt(file_path, skip_header=14,skip_footer = 6, usecols=(7))
    data = data[~np.isnan(data)]
    baseline_bperp_values.append(np.median(data))
    
    date = os.path.basename(file_path)[9:17]
    date_objects_baseline.append(datetime.strptime(date, '%Y%m%d'))
    date_strings_baseline.append(date)
    data = None

plt.figure(figsize=(10, 6)) #Plots baseline values throughout time period
plt.plot(date_objects_baseline, baseline_bperp_values,marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Baseline Intensity')
plt.title('Background signal')
# Format the x-axis to show fewer date labels
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# Rotate date labels for better readability
plt.gcf().autofmt_xdate()
# Grid for better visualization
plt.grid(True)
# Save the plot
plt.savefig(baseline_plot_path)
plt.close()

baseline_data = np.asarray(np.stack((date_strings_baseline,baseline_bperp_values),axis = 1))
# save to csv file
np.savetxt(baseline_data_path, baseline_data, delimiter=',', fmt='%s')


################Read low baseline backscatter values################

dates = []
pth = glob.glob(os.path.join(raw_images_path,raw_tif_path_format[0]+'*'+raw_tif_path_format[1])) #Get all those sweet sweet paths (from raw images)
pth.sort()
pth_array = np.array(pth)
for p in pth:
	dates.append(os.path.basename(p)[start_of_date:start_of_date+8])
	
i=0
img_wrongsize = []
img_shapes = []

for p in pth:
    img = rio.open(p).read(1)    ##before p was os.path.join( raw_images_path, raw_tif_path_format[0]+dates[i]+raw_tif_path_format[1] )
    array = np.array(img)  
    if i == 0:
        shape_firstimg = array.shape
    if array.shape != shape_firstimg:   #(676,980): is normally this shape
        img_wrongsize.append(i)
    i += 1

resize_and_save_images(dates, img_wrongsize, folder_tosave_name = resized_backscatter_images, 
                       raw_images_path = raw_images_path, target_shape=shape_firstimg,path_format = raw_tif_path_format)

resized_paths = glob.glob(os.path.join(resized_backscatter_images,raw_tif_path_format[0]+'*'+raw_tif_path_format[1])) #Get all those sweet sweet paths
resized_paths.sort()
resized_paths_array = np.array(resized_paths)

resized_dates = []
for path in resized_paths:
	resized_dates.append(os.path.basename(path)[start_of_date:start_of_date+8])

baseline_bperp_values_array = np.array(baseline_bperp_values)
date_strings_baseline_array = np.array(date_strings_baseline)

low_baseline_dates = date_strings_baseline_array[abs(baseline_bperp_values_array) <= max_baseline]   
bool_good_dates_from_baseline = np.isin(resized_dates, low_baseline_dates)   
low_baseline_paths = resized_paths_array[bool_good_dates_from_baseline]


####Open example TIF for geotransform####
geotiff = gdal.Open(example_path_for_geotransform) #TSX MASTER
data = geotiff.ReadAsArray()

# Boundaries of tif
ulx, xres, xskew, uly, yskew, yres = geotiff.GetGeoTransform()
lrx = ulx + (geotiff.RasterXSize * xres)
lry = uly + (geotiff.RasterYSize * yres)
extent=[ulx, lrx, lry, uly]
Y_pixel_gdal = geotiff.RasterYSize
X_pixel_gdal = geotiff.RasterXSize


image_file = rio.open(example_path_for_geotransform).read(1)
img = np.array(image_file)
x_pixels, y_pixels = img.shape

####Plot backscatter differences and save them as specified at start of the script

frames_per_avg = 1
for p in range(frames_per_avg, len(low_baseline_paths) - frames_per_avg + 1): #Plots the backscatter ratio change of ALL images, and also the backscatter absolute change of all images
    #                                                                           within the specified baseline window!
    #                                                                           (Also +1 so you dont get the last one but just up to it. )
    comparison = last_n_avg(low_baseline_paths[p-frames_per_avg:p]) #Only gives up to p-1'th value
    
    comparee = rio.open(os.path.join(resized_backscatter_images, raw_tif_path_format[0]+low_baseline_dates[p]+raw_tif_path_format[1])).read(1)
    comparee[comparee <= 0.] = np.NaN
    comparee = 10 * np.log10(comparee)

    r0dB_ratio = 2*(comparee-comparison)/(comparee+comparison) #Multiply by 2 from Goitom et al 2015
    r0dB_absolute = comparee - comparison
        
        
    plot_change(r0dB_ratio, os.path.join(ratio_change, f"{low_baseline_dates[p]}_backscatter_ratchange"), 
                low_baseline_dates[p],range=[-1,1],geotiff = geotiff,extent=[ulx, lrx, lry, uly])
    
    r0dB_ratio_reducednoise = r0dB_ratio.copy()
    upper_percentile = np.nanpercentile(r0dB_ratio_reducednoise.flatten(),100 - percentile_selection)
    lower_percentile = np.nanpercentile(r0dB_ratio_reducednoise.flatten(), percentile_selection)
    r0dB_ratio_reducednoise[np.where((r0dB_ratio_reducednoise>=lower_percentile) & (r0dB_ratio_reducednoise<=upper_percentile))] = np.NaN
    plot_change(r0dB_ratio_reducednoise, os.path.join(ratio_change_inpercentile, f"{low_baseline_dates[p]}_backscatter_ratchange_rednoise"), 
                low_baseline_dates[p],range=[-1,1],geotiff = geotiff,extent=[ulx, lrx, lry, uly])
        
    # plot_change(r0dB_absolute, os.path.join(absolute_change, f"{low_baseline_dates[p]}_backscatter_abschange"), 
    #             low_baseline_dates[p],range =[np.min(r0dB_absolute.flatten()),np.max(r0dB_absolute.flatten())],
    #             geotiff = geotiff,extent=[ulx, lrx, lry, uly])  
    
    

##################################Linear Inversion begins here##################################
    
#low_baseline_dates is the variable with the date strings which we can now us
#To do inversion I will choose the 3 images before an event and the one straight after 
#convert list of date strings to date objects

events_objs = []
for date in events:
    events_objs.append(datetime.strptime(date, '%Y%m%d'))

low_baseline_dates_objs = []
for date in low_baseline_dates:
    low_baseline_dates_objs.append(datetime.strptime(date, '%Y%m%d'))
low_baseline_dates_objs = np.array(low_baseline_dates_objs)

#################Find best prev_event_dates and post_event_dates ###################

indices_after_events = []

for event_ob in events_objs:
    indices_after_events.append(np.where(low_baseline_dates_objs > event_ob)[0][0])     #INDEX of the first true value in the array

differences = np.append(np.diff(indices_after_events),len(low_baseline_dates_objs)-indices_after_events[-1])

####This block alters number of images before event to include in lin inversion based on how close together they are
events_n_before = []
for idx, event_index in enumerate(indices_after_events):
    if idx == 0:
        bounds_lower = int(event_index) if event_index < 3 else steps_prev_default
        
        if differences[idx] < 3: #I.e. if there is less than 2 images taken between 2 events the inversion may still include tefra from the previous eruption event!
            warnings.warn(f"The events on {events[idx-1]}, {events[idx]} do not have enough data between them to give reliable Linear Inversion because they overlap")
        
    bounds_lower = int(differences[idx-1]) if differences[idx-1] < 3 else steps_prev_default
    
    if differences[idx] < 3: 
            warnings.warn(f"The events on {events[idx-1]}, {events[idx]} do not have enough data between them to give reliable Linear Inversion!")
    
    events_n_before.append(bounds_lower)

####This block defines slices the dates array to define which dates will be in each events' linear inversion
events_dates = []
for idx, event_ob in enumerate(events_objs):
    index = indices_after_events[idx]
    bounds_lower = events_n_before[idx]
    lower_bound_index = index - bounds_lower
    upper_bound_index = index
    events_dates.append(low_baseline_dates[lower_bound_index:upper_bound_index + 1])

###This block loads the images into an array so they can be processed efficiently in lin inversion
events_images = []
for idx,event_ob in enumerate(events_objs):
    particular_event_images = []
    for jdx in range(len(events_dates[idx])):      
        img = rio.open(os.path.join(resized_backscatter_images, raw_tif_path_format[0]+events_dates[idx][jdx]+raw_tif_path_format[1])).read(1)
        img[img<= 0] = np.NaN 
        img = 10*np.log10(img)
        particular_event_images.append(img)    
    events_images.append(particular_event_images)
    
    
######################Linear inversion code#####################

#This is the method Pedro reccomended for linear inversion

for idx,particular_event_images in enumerate(events_images):
    
    particular_event_nparray_images = np.array(particular_event_images)
    no_observations = len(particular_event_nparray_images)

    # Construct the design matrix G
    G = np.zeros((no_observations, 2))
    G[:, 1] = 1     ########################### Assuming the step occurs after the first image ##########################
    G[-1,0] = 1
    
    regularization_term = 1e-12
    
    ########################## Solve for the model parameters m using least squares##########################
    GTG_inv = np.linalg.inv(G.T @ G) + (regularization_term *  np.eye(2))
    
    reshaped_images = particular_event_nparray_images.reshape(no_observations, -1)
    
    m_flat = GTG_inv @ G.T @ reshaped_images
    
    m = m_flat.reshape(2, x_pixels, y_pixels)
    
    # Extract the pre-step level and step estimation
    pre_step_level = m[1]
    step_estimation = m[0]
    
    backscatter_ratio_image = 2 * step_estimation / ((2 * pre_step_level) + step_estimation) 

    plot_change(backscatter_ratio_image, os.path.join(LinInversionPath, 
                f"Linear_inversion_{events[idx]}"), 
                date = events[idx],range=[-1,1],geotiff = geotiff, extent=extent)
    
    backscatter_ratio_image_reducednoise = backscatter_ratio_image.copy()
    
    upper_percentile = np.nanpercentile(backscatter_ratio_image_reducednoise.flatten(),100 - percentile_selection)
    lower_percentile = np.nanpercentile(backscatter_ratio_image_reducednoise.flatten(), percentile_selection)
    backscatter_ratio_image_reducednoise[np.where((backscatter_ratio_image_reducednoise>=lower_percentile) & 
                                                  (backscatter_ratio_image_reducednoise<=upper_percentile))] = np.NaN
    
    plot_change(backscatter_ratio_image_reducednoise, 
                os.path.join(LinInversionPathReduced, f"Linear_inversion_reducednoise{events[idx]}"), 
                date = events[idx],range=[-1,1],geotiff = geotiff, extent=extent)
    
    
######################Formatting data for clustering

additional_tif_feature_paths = [os.path.join(DEM_folder_path,DEM_filename), os.path.join(DEM_folder_path,aspect_filename),
                                os.path.join(DEM_folder_path,roughness_filename),os.path.join(DEM_folder_path,slope_filename)]



linear_inversions_paths = glob.glob(os.path.join(LinInversionPathReduced,'*.tif'))
linear_inversions_paths.sort()

valid_mask = ~np.isnan(image_file)
valid_mask = valid_mask
#dates = events


num_imgs = len(events) #have we already got this?
# Initialize the array for linear inversion data
linear_inversion_arrays = np.zeros((num_imgs, Y_pixel_gdal, X_pixel_gdal))

# Read and store linear inversion data without transposition
for idx, file_path in enumerate(linear_inversions_paths):
    with rio.open(file_path) as src:
        linear_inversion_arrays[idx, :, :] = src.read(1)  # No transpose here

# Apply valid mask (no transposition needed)
valid_data = linear_inversion_arrays[:, valid_mask].T
valid_data[np.isnan(valid_data)] = 0.

## Add x and y coordinates to feature space
x_coords, y_coords = np.meshgrid(np.arange(X_pixel_gdal), np.arange(Y_pixel_gdal))
x_coords_valid = x_coords[valid_mask]
y_coords_valid = y_coords[valid_mask]

coords_valid = np.column_stack((x_coords_valid, y_coords_valid))
data_with_coords = np.column_stack((valid_data, coords_valid))

# Initialize array for additional features
num_additional_features = len(additional_tif_feature_paths)
additional_features_arrays = np.zeros((num_additional_features, Y_pixel_gdal, X_pixel_gdal))

# Read and store additional feature data without transposition
for idx, file_path in enumerate(additional_tif_feature_paths):
    with rio.open(file_path) as src:
        additional_features_arrays[idx, :, :] = src.read(1)  # No transpose here

# Apply valid mask to additional features (no transposition needed)
additional_features_valid = []
for idx in range(num_additional_features):
    feature_valid = additional_features_arrays[idx, :, :][valid_mask]
    additional_features_valid.append(feature_valid)

# Combine valid data and additional features
additional_features_valid = np.column_stack(additional_features_valid)
data_with_valid_features = np.column_stack((valid_data, additional_features_valid))

# Scale data
scaler = preprocessing.StandardScaler().fit(data_with_valid_features)
scaled_data = scaler.transform(data_with_valid_features)

# Apply custom weightings
scaled_data[:, -1] *= slope_weighting    # Slope from DEM
scaled_data[:, -2] *= roughness_weighting
scaled_data[:, -3] *= aspect_weighting 
scaled_data[:, -4] *= DEM_weighting      # DEM
scaled_data[:, -7:-5] *= coordinate_weighting

# Convert to numpy array and save
data = np.asarray(scaled_data)
np.savetxt(where_to_save_clusterdata, data, delimiter=',')

############## Clustering Data ###############
for idx, cluster_path in enumerate(cluster_paths):
    seed = seeds[idx]
    kmeans_loss = []

    for num_clusters in range(cluster_min, cluster_max):
        kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
        kmeans.fit(scaled_data)
        kmeans_loss.append(kmeans.inertia_)
        clustered_labels = kmeans.labels_

        # Reshape the clustering result back to the original image shape
        clustered_image = np.full((Y_pixel_gdal, X_pixel_gdal), np.nan)
        clustered_image[valid_mask] = clustered_labels

        if num_clusters in clusters_to_plot_separate:
            for cluster_num in range(num_clusters):
                cluster_only_image = np.full(( Y_pixel_gdal, X_pixel_gdal), np.nan)
                cluster_only_image[valid_mask] = np.where(clustered_labels == cluster_num, cluster_num, np.nan)

                # Create a custom colormap with 'gray' for NaN values and 'red' for the current cluster
                colors = ['gray'] * num_clusters
                colors[cluster_num] = 'red'
                custom_cmap = ListedColormap(colors)

                # Plot the image with only the current cluster
                plt.figure(figsize=(10, 10))
                plt.imshow(cluster_only_image, cmap=custom_cmap, vmin=0, vmax=num_clusters - 1)
                plt.title(f"Cluster {cluster_num} Only - KMeans with {num_clusters} Clusters")
                plt.savefig(os.path.join(cluster_plot_separate[idx][clusters_to_plot_separate.index(num_clusters)], f"Cluster_{cluster_num}_of_{num_clusters}_clusters_kmeans.png"), 
                            dpi=300, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close()

        palette = sns.color_palette('husl', num_clusters)
        clust_im = np.full(( Y_pixel_gdal, X_pixel_gdal, 3), np.nan)
        colors = np.array(palette)[clustered_labels]
        clust_im[valid_mask] = colors

        nan_color = [0, 0, 0]  # Black color
        clust_im = np.where(np.isnan(clust_im), nan_color, clust_im)

        # Plot the clustered image
        plt.figure(figsize=(10, 10))
        plt.imshow(clust_im)

        # Create a legend with cluster labels
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10, label=f'Cluster {i}')
                   for i in range(num_clusters)]
        plt.legend(handles=handles, loc='upper right')
        plt.title(f"KMeans Clustering with {num_clusters} Clusters")
        plt.savefig(os.path.join(cluster_path, f"{num_clusters}_clusters_kmeans_speckle{speckle_extent}.png"), 
                    dpi=300, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Plot KMeans loss for different cluster counts
    plt.figure(figsize=(10, 10))
    plt.title("KMeans loss for different cluster count")
    plt.xlabel("Number of Clusters", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.plot(np.arange(cluster_min, cluster_max), kmeans_loss, marker='o', linestyle='-', color='b')

    # Save the loss plot
    output_path = os.path.join(cluster_path, "lossplot.png")
    plt.savefig(output_path, dpi=300, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()













# linear_inversion_arrays = np.zeros((num_imgs, y_pixels, x_pixels))

# # Read and store linear inversion data with transposition
# for idx, file_path in enumerate(linear_inversions_paths):
#     with rio.open(file_path) as src:
#         linear_inversion_arrays[idx, :, :] = src.read(1).T  # Transpose here to align dimensions

# # Apply valid mask and transpose
# valid_data = linear_inversion_arrays[:, valid_mask.T].T
# valid_data[np.isnan(valid_data)] = 0.

# ## Add x and y coordinates to feature space
# x_coords, y_coords = np.meshgrid(np.arange(x_pixels), np.arange(y_pixels))
# x_coords_valid = x_coords[valid_mask.T]
# y_coords_valid = y_coords[valid_mask.T]

# coords_valid = np.column_stack((x_coords_valid, y_coords_valid))
# data_with_coords = np.column_stack((valid_data, coords_valid))

# # Initialize array for additional features
# num_additional_features = len(additional_tif_feature_paths)
# additional_features_arrays = np.zeros((num_additional_features, y_pixels, x_pixels))

# # Read and store additional feature data
# for idx, file_path in enumerate(additional_tif_feature_paths):
#     with rio.open(file_path) as src:
#         additional_features_arrays[idx, :, :] = src.read(1).T  # Transpose here for consistency

# # Apply valid mask to additional features
# additional_features_valid = []
# for idx in range(num_additional_features):
#     feature_valid = additional_features_arrays[idx, :, :][valid_mask.T]
#     additional_features_valid.append(feature_valid)

# # Combine valid data and additional features
# additional_features_valid = np.column_stack(additional_features_valid)
# data_with_valid_features = np.column_stack((valid_data, additional_features_valid))

# # Scale data
# scaler = preprocessing.StandardScaler().fit(data_with_valid_features)
# scaled_data = scaler.transform(data_with_valid_features)

# # Apply custom weightings
# scaled_data[-1] *= slope_weighting    # Slope from DEM
# scaled_data[-2] *= roughness_weighting
# scaled_data[-3] *= aspect_weighting 
# scaled_data[-4] *= DEM_weighting      # DEM
# scaled_data[-7:-5] *= coordinate_weighting

# # Convert to numpy array and save
# data = np.asarray(scaled_data)
# np.savetxt(where_to_save_clusterdata, data, delimiter=',')

# ############## Clustering Data ###############
# for idx, cluster_path in enumerate(cluster_paths):
#     seed = seeds[idx]
#     kmeans_loss = []

#     for num_clusters in range(cluster_min, cluster_max):
#         kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
#         kmeans.fit(scaled_data)
#         kmeans_loss.append(kmeans.inertia_)
#         clustered_labels = kmeans.labels_

#         # Reshape the clustering result back to the original image shape
#         clustered_image = np.full((y_pixels, x_pixels), np.nan)
#         clustered_image[valid_mask.T] = clustered_labels

#         if num_clusters in clusters_to_plot_separate:
#             for cluster_num in range(num_clusters):
#                 cluster_only_image = np.full((y_pixels, x_pixels), np.nan)
#                 cluster_only_image[valid_mask.T] = np.where(clustered_labels == cluster_num, cluster_num, np.nan)

#                 # Create a custom colormap with 'gray' for NaN values and 'red' for the current cluster
#                 colors = ['gray'] * num_clusters
#                 colors[cluster_num] = 'red'
#                 custom_cmap = ListedColormap(colors)

#                 # Plot the image with only the current cluster
#                 plt.figure(figsize=(10, 10))
#                 plt.imshow(cluster_only_image, cmap=custom_cmap, vmin=0, vmax=num_clusters - 1)
#                 plt.title(f"Cluster {cluster_num} Only - KMeans with {num_clusters} Clusters")
#                 plt.savefig(os.path.join(cluster_plot_separate[idx][clusters_to_plot_separate.index(num_clusters)], f"Cluster_{cluster_num}_of_{num_clusters}_clusters_kmeans.png"), 
#                             dpi=300, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
#                 plt.close()

#         palette = sns.color_palette('husl', num_clusters)
#         clust_im = np.full((y_pixels, x_pixels, 3), np.nan)
#         colors = np.array(palette)[clustered_labels]
#         clust_im[valid_mask.T] = colors

#         nan_color = [0, 0, 0]  # Black color
#         clust_im = np.where(np.isnan(clust_im), nan_color, clust_im)

#         # Plot the clustered image
#         plt.figure(figsize=(10, 10))
#         plt.imshow(clust_im)

#         # Create a legend with cluster labels
#         handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10, label=f'Cluster {i}')
#                    for i in range(num_clusters)]
#         plt.legend(handles=handles, loc='upper right')
#         plt.title(f"KMeans Clustering with {num_clusters} Clusters")
#         plt.savefig(os.path.join(cluster_path, f"{num_clusters}_clusters_kmeans_speckle{speckle_extent}.png"), 
#                     dpi=300, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
#         plt.close()

#     # Plot KMeans loss for different cluster counts
#     plt.figure(figsize=(10, 10))
#     plt.title("KMeans loss for different cluster count")
#     plt.xlabel("Number of Clusters", fontsize=14)
#     plt.ylabel("Loss", fontsize=14)
#     plt.plot(np.arange(cluster_min, cluster_max), kmeans_loss, marker='o', linestyle='-', color='b')

#     # Save the loss plot
#     output_path = os.path.join(cluster_path, "lossplot.png")
#     plt.savefig(output_path, dpi=300, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
#     plt.close()








