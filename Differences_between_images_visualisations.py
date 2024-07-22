import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import rasterio as rio
from datetime import datetime
import matplotlib.dates as mdates

directorio = os.getcwd()


TIF_image_paths = glob.glob(f"{directorio}/resized_images/*.tif") #Makes sure all images are same size, this folder has come from another 
TIF_image_paths.sort()  #I didnt have this line and got errors for so long hahaha

TIF_image_dates = []
date_strings = []
for idx, TIF_image_date in enumerate(TIF_image_paths):
    date_strings.append(os.path.basename(TIF_image_date)[8:16])
    TIF_image_dates.append(datetime.strptime(date_strings[idx], '%Y%m%d'))

TIF_image_dates_array = np.array(TIF_image_dates)

date_differences = np.diff(TIF_image_dates_array)

date_differences_days = np.array([diff.days for diff in date_differences])

plt.figure(figsize=(10, 6))
plt.hist(date_differences_days, bins=100, edgecolor='black')
plt.xlabel('Days between images')
plt.ylabel('Frequency')
plt.title('Histogram of Differences Between Image Dates')
plt.grid(True)

# Save the histogram plot
plt.savefig(f'{directorio}/Differences_between_images_visualisations.png')
plt.close()


# Remove the last date to match the shape of date differences
dates_droplast = TIF_image_dates_array[:-1]

# Plot the differences over time
plt.figure(figsize=(10, 6))
plt.plot(dates_droplast, date_differences_days, marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Days between images')
plt.title('Difference Between Image Dates')


# Format the x-axis to show fewer date labels
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Rotate date labels for better readability
plt.gcf().autofmt_xdate()

# Grid for better visualization
plt.grid(True)

# Save the plot
plt.savefig(f'{directorio}/Differences_between_images_dates.png')
plt.close()




