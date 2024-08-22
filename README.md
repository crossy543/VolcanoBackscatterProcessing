# VolcanoBackscatterProcessing
Some code I have written for a research project to automate some of the useful image processing techniques when analysing SAR backscatter changes

The methods are described in detail in the unfinished report which is in the repository.

The main script here is BackscatterAnalysis.py. The other two scripts are useful for exploring the SAR imagery but are not a part of the main pipeline in backscatter analysis. 

This script follows the flow below, which I will modify later on when the code has been fully updated. 

This script has around 100 lines at the start where you will need to choose your variables and enter filenames. The only file inputs which are needed is the actual SAR images, with a speckle filter already applied, the baseline backscatter values and the DEM feature which you want to give to the clustering algorithm in order to achieve better results. 

After you have moved all of these files into a folder, and named that folder in the script where it is required along with the paths to this file, you can run the code (I have just used terminal but equally VSCode or spyder command lines will work) and you will have a new folder downloaded, which has the below attributes. 
<img width="1174" alt="Screenshot 2024-08-22 at 16 50 35" src="https://github.com/user-attachments/assets/2adc6dbb-7fa4-4046-b87c-ad0b682035c0">

The flow of the code is as below, it is not fully up to date but it will be when the code is completely finished.

<img width="843" alt="image" src="https://github.com/user-attachments/assets/05606fc0-bdd1-4485-a2ac-9da96e1944eb">
