import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import rasterio as rio




directorio= os.getcwd()
print("You working at directory path:" + directorio)


img = rio.open(f'{directorio}/TIF/conv_EQA20200103.isp.mli.geo.tif').read(1)
img[img == 0] = np.nan
img = 10 * np.log10(img)


# Display the image
fig, ax = plt.subplots()
ax.imshow(img)

# Function to capture the click event
coords = []

def onclick(event):
    ix, iy = event.xdata, event.ydata
    coords.append((int(ix), int(iy)))
    print(f'Coordinates: x={ix}, y={iy}')
    if len(coords) == 1:
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)

# Connect the click event to the function
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

while not coords:
    plt.pause(0.1)

# Coordinates will be stored in the coords list
print(f'Selected coordinates: {coords}')

x,y = coords[0]

#Numerate the files
dates = []
pth = glob.glob(f"{directorio}/TIF/*.tif")
pth.sort()
for p in pth:
	date = os.path.basename(p)[8:16]
	dates.append(date)
number_images = len(dates)

pixel_avg_change = os.path.join(directorio, f"pixel_{x}_{y}_profile")
os.makedirs(pixel_avg_change, exist_ok=True)
print(pixel_avg_change)

def calculate_avg_intensity(img, x, y,n_smoothing):
    neighbours = img[max(0, y-n_smoothing):y+n_smoothing+1, max(0, x-n_smoothing):x+n_smoothing+1]
    
    neighbours[neighbours == 0] = np.nan
    px_cluster = 10 * np.log10(neighbours)
    
    return np.nanmean(px_cluster)


px_val = np.empty(number_images)

for j in range(0,3): #Loops over to chsnge n_smoothing so we can see how it changes having an increasing number of surrounding pixels in avg calculation to be sure of any trends
    
    if j == 0: #Returns plot of SAR map with dot as chosen point
        image_file = rio.open(f"{directorio}/TIF/conv_EQA20200205.isp.mli.geo.tif").read(1)
        img = image_file 
        
        img[img == 0] = np.nan
        r0dB = 10 * np.log10(img)
        
        fig, ax = plt.subplots()
        ax.imshow(r0dB, cmap='gray')
        ax.plot(x, y, 'ro')  # Plot the selected pixel as a red dot
        ax.set_title(f'Selected Pixel at ({x}, {y})')
        plt.savefig(f'{pixel_avg_change}/pixmap.jpg', dpi=300, format='jpg', transparent='true', bbox_inches='tight')
        plt.close()
    
    for i, date in enumerate(dates):
        image_file = rio.open(f"{directorio}/TIF/conv_EQA{date}.isp.mli.geo.tif").read(1)
        img = image_file

        
        
        px_val[i] = calculate_avg_intensity(img, x, y,j)      
        
    interval_indices = range(0, len(dates), 6)

    plt.figure()
    plt.plot(dates,px_val,marker = 'o')
    plt.xlabel('Time')
    plt.ylabel('Intensity (dB)')
    plt.title(f'Intensity of Erosion Channel Pixel Over Time Smoothed over {j} pixels')

    plt.xticks([dates[i] for i in interval_indices], rotation=45)

    plt.tight_layout()
    plt.savefig(f'{pixel_avg_change}/smoothed{j}_pix.jpg', format='jpg', transparent='true', bbox_inches='tight')
    plt.close()