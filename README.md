# VolcanoBackscatterProcessing
Some code I have written for a research project to automate some of the useful image processing techniques when analysing SAR backscatter changes


The main script here is BackscatterAnalysis.py. The other two scripts are useful for exploring the SAR imagery but are not a part of the main pipeline in backscatter analysis. 

This script follows the flow below, which I will modify later on when the code has been fully updated. 

This script has around 100 lines at the start where you will need to choose your variables and enter filenames. The only file inputs which are needed is the actual SAR images, with a speckle filter already applied, the baseline backscatter values and the DEM feature which you want to give to the clustering algorithm in order to achieve better results. 

After you have moved all of these files into a folder, and named that folder in the script where it is required along with the paths to this file, you can run the code (I have just used terminal but equally VSCode or spyder command lines will work) and you will have a new folder downloaded, which is shown in the plot: 
