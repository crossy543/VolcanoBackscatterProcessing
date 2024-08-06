import os
import glob
import numpy as np
from datetime import datetime
import shutil

directorio= os.getcwd()
speckle_string = "speckle3x3"
speckle_path = os.path.join(directorio + '/Speckle_Filter_Applied3x3')

all_backscatter_images = os.path.join(speckle_path + f'/All_backscatter_images_{speckle_string}')
lowbaseline_backscatter_images = os.path.join(speckle_path + f'/Low_baseline_backscatter_images_{speckle_string}')
resized_images = os.path.join(speckle_path + f'/resized_images_nobaseline_{speckle_string}')

chosen_images = all_backscatter_images

specific_events = os.path.join(f"{chosen_images}_SpecificEvents")  #Where images are saved to!
os.makedirs(specific_events, exist_ok=True)

events = ["20200608","20200920","20210305","20210311","20210412","20210507","20211212","20220404","20231202"]
events_objs = []

for idx,date in enumerate(events):
    events_objs.append(datetime.strptime(date, '%Y%m%d'))
    

TIF_image_paths = glob.glob(f"{chosen_images}/*.tif") 
TIF_image_paths.sort()  #I didnt have this line and got errors for so long hahaha

TIF_image_dates = []
date_strings = []

for idx, TIF_image_date in enumerate(TIF_image_paths):
    date_strings.append(os.path.basename(TIF_image_date)[0:8])
    TIF_image_dates.append(datetime.strptime(date_strings[idx], '%Y%m%d'))

TIF_image_dates_array = np.array(TIF_image_dates)

indices_after_events = []
for event_ob in events_objs:
    indices_after_events.append(np.where(TIF_image_dates_array > event_ob)[0][0])  


for idx in indices_after_events:
    ##Save event_path file to specific_events folder
    destination_path = os.path.join(specific_events, os.path.basename(TIF_image_paths[idx]))
    shutil.copy(TIF_image_paths[idx], destination_path)  # Copy the file to the specific_events folder