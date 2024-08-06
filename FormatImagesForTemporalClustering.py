import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import math
import rasterio as rio
from osgeo import gdal
from sklearn.cluster import KMeans
from sklearn import preprocessing
from osgeo import osr
from matplotlib.colors import ListedColormap
from numpy import asarray
from numpy import savetxt




def calculate_distance_from_center(geo_transform, rows, cols, center_coords=(-78.3273, -2.0142)):
    """Calculate the distance of each pixel from the given center coordinates."""
    ulx, xres, xskew, uly, yskew, yres = geo_transform

    col_indices, row_indices = np.meshgrid(np.arange(cols), np.arange(rows), indexing='xy')

    x_coords = ulx + (col_indices * xres) + (row_indices * xskew)
    y_coords = uly + (col_indices * yskew) + (row_indices * yres)
    
    distances = np.sqrt((x_coords - center_coords[0])**2 + (y_coords - center_coords[1])**2)
    
    return distances


def center_unit_mean_and_variance(data):
    scaler = preprocessing.StandardScaler().fit(data)
    return scaler.transform(data)


directorio = os.getcwd()


################### All backscatter images ######################


glob_backscatter_plots_nospeckle = glob.glob(directorio + '/All_backscatter_images/*.tif')
glob_backscatter_plots_nospeckle.sort()

dates = [os.path.basename(p)[5:13] for p in glob_backscatter_plots_nospeckle]

first_image = gdal.Open(glob_backscatter_plots_nospeckle[0])
Y_pixel_nospeckle = first_image.RasterYSize
X_pixel_nospeckle = first_image.RasterXSize
geo_transform_gdal = first_image.GetGeoTransform()
first_image = None

num_imgs_nospeckle = len(dates)
backscatter_plots_arrays_nospeckle = np.zeros((num_imgs_nospeckle, Y_pixel_nospeckle, X_pixel_nospeckle))

for idx, file_path in enumerate(glob_backscatter_plots_nospeckle):
    with rio.open(file_path) as src:
        backscatter_plots_arrays_nospeckle[idx, :, :] = src.read(1)
    
# # Create a mask for valid data points (where there are no NaNs in the time series)
valid_mask = ~np.isnan(backscatter_plots_arrays_nospeckle).any(axis=0)

#################### Linear Inversion of Events #################

glob_backscatter_plots = glob.glob(directorio + '/Speckle_Filter_Applied3x3/All_backscatter_images_speckle3x3_SpecificEvents/*.tif')
glob_backscatter_plots.sort()

dates = [os.path.basename(p)[0:8] for p in glob_backscatter_plots]

first_image = gdal.Open(glob_backscatter_plots[0])
Y_pixel = first_image.RasterYSize
X_pixel = first_image.RasterXSize
geo_transform_gdal = first_image.GetGeoTransform()
first_image = None

num_imgs = len(dates)
backscatter_plots_arrays = np.zeros((num_imgs, Y_pixel, X_pixel))

for idx, file_path in enumerate(glob_backscatter_plots):
    with rio.open(file_path) as src:
        backscatter_plots_arrays[idx, :, :] = src.read(1)
    
    
valid_data = backscatter_plots_arrays[:, valid_mask].T
valid_data[np.isnan(valid_data)] = 0.

print(len(valid_mask[valid_mask==False]),len(valid_mask[valid_mask==True]) )          


distances = calculate_distance_from_center(geo_transform_gdal, Y_pixel, X_pixel)    

valid_distance = distances[valid_mask].reshape(-1,1)

data_with_valid_distance = np.column_stack((valid_data, valid_distance))


additional_tif_feature_paths = [f'{directorio}/DEM_downloads/FullDEM/DEM.tif', f'{directorio}/DEM_downloads/FullDEM/viz.hh_aspect.tif',
                                f'{directorio}/DEM_downloads/FullDEM/viz.hh_roughness.tif',f'{directorio}/DEM_downloads/FullDEM/viz.hh_slope.tif']

# Initialize array for additional features
num_additional_features = len(additional_tif_feature_paths)
additional_features_arrays = np.zeros((num_additional_features, Y_pixel, X_pixel))

# Read and store additional feature data
for idx, file_path in enumerate(additional_tif_feature_paths):
    with rio.open(file_path) as src:
        additional_features_arrays[idx, :, :] = src.read(1) 
        
additional_features_valid = []
for idx in range(num_additional_features):
    feature_valid = additional_features_arrays[idx, :, :][valid_mask]
    additional_features_valid.append(feature_valid)
    

additional_features_valid = np.column_stack(additional_features_valid)


data_with_valid_features = np.column_stack((data_with_valid_distance, additional_features_valid))

scaled_data= center_unit_mean_and_variance(data_with_valid_features)

print(scaled_data.shape)
##################################increased weighting on distance ###############################
#scaled_data[len(scaled_data)-num_additional_features-1:] *= 20
scaled_data[-4] *= 20 #DEM
scaled_data[-5] *= 5 #Distance from center
scaled_data[-1] *= 10 #Slope from DEM

cluster_path = os.path.join(directorio, "Clustering_formatted_temporal")
os.makedirs(cluster_path,exist_ok = True)

data = asarray(scaled_data)
# save to csv file
savetxt(f'{cluster_path}/speckle3x3_bigevents_clustering_temporaldata.csv', data, delimiter=',')