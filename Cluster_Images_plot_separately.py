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
from numpy import genfromtxt


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


# glob_backscatter_plots = glob.glob(directorio + '/All_backscatter_images/*.tif')
# glob_backscatter_plots.sort()

# dates = [os.path.basename(p)[5:13] for p in glob_backscatter_plots]



#################### Linear Inversion of Events #################

glob_backscatter_plots = glob.glob(directorio + '/LinearInversionsForSpecificEvents/*.tif')
glob_backscatter_plots.sort()

dates = [os.path.basename(p)[5:13] for p in glob_backscatter_plots]

######################################################

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
    
# Create a mask for valid data points (where there are no NaNs in the time series)
valid_mask = ~np.isnan(backscatter_plots_arrays).any(axis=0)
valid_data = backscatter_plots_arrays[:, valid_mask].T

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


##################################increased weighting on distance ###############################
#scaled_data[len(scaled_data)-num_additional_features-1:] *= 20
scaled_data[-4] *= 20 #DEM
scaled_data[-5] *= 5 #Distance from center
scaled_data[-1] *= 10 #Slope from DEM



#n = 1

cluster_path = os.path.join(directorio, "Clustering_figures")
os.makedirs(cluster_path,exist_ok = True)
os.makedirs(os.path.join(cluster_path, "Clustering_DEM_Features_indiviudal_clusters"),exist_ok = True)
#os.makedirs(os.path.join(cluster_path, f"Clustering_DEM_Features_weighted4/image_{n}"),exist_ok = True)

all_images_clustering_temporaldata = genfromtxt(f'{directorio}/Clustering_formatted_temporal/all_images_clustering_temporaldata.csv')

kmeans_loss = []
num_clusters = 6

kmeans = KMeans(n_clusters=num_clusters, random_state=11)
kmeans.fit(scaled_data, sample_weight=None)
kmeans_loss.append(kmeans.inertia_)
clustered_labels = kmeans.labels_

# Reshape the clustering result back to the original image shape
clustered_image = np.full((Y_pixel, X_pixel), np.nan)
clustered_image[valid_mask] = clustered_labels

cluster_centers = kmeans.cluster_centers_
valid_coords = np.argwhere(valid_mask)

cluster_centers_2d = []


for center in cluster_centers:
    # Only consider the valid data for comparison
    valid_data_with_distance = data_with_valid_distance[:, :data_with_valid_distance.shape[1] - 1]  # exclude distance for comparison
    distances_to_center = np.linalg.norm(valid_data_with_distance - center[:valid_data_with_distance.shape[1]], axis=1)
    nearest_idx = np.argmin(distances_to_center)
    cluster_centers_2d.append(valid_coords[nearest_idx])

# Create separate images for each cluster
for cluster_num in range(num_clusters):
    print('done')
    cluster_only_image = np.full((Y_pixel, X_pixel), np.nan)
    cluster_only_image[valid_mask] = np.where(clustered_labels == cluster_num, cluster_num, np.nan)

    # Create a custom colormap with 'gray' for NaN values and 'red' for the current cluster
    colors = ['gray'] * num_clusters
    colors[cluster_num] = 'red'  # Highlight the current cluster in red
    custom_cmap = ListedColormap(colors)

    # Plot the image with only the current cluster
    plt.figure(figsize=(10, 10))
    plt.imshow(cluster_only_image, cmap=custom_cmap, vmin=0, vmax=num_clusters - 1)

    plt.title(f"Cluster {cluster_num} Only - KMeans with {num_clusters} Clusters")
    plt.savefig(os.path.join(cluster_path, "Clustering_DEM_Features_indiviudal_clusters", f"Cluster_{cluster_num}_of_{num_clusters}_clusters_kmeans.png"), 
                dpi=300, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    



# clustered_image_path = os.path.join(directorio, 'clustered_image.tif')
# driver = gdal.GetDriverByName('GTiff')
# out_ds = driver.Create(clustered_image_path, cols, rows, 1, gdal.GDT_UInt16)

# # Save the clustered image
# geo_transform, crs = get_geo_transform(glob_backscatter_plots[0])
# out_ds.SetGeoTransform(geo_transform)
# out_ds.SetProjection(crs.to_wkt())

# out_ds.GetRasterBand(1).WriteArray(clustered_image)
# out_ds.FlushCache()
# out_ds = None  # Close the file


    











