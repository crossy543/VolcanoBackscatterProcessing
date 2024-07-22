# VolcanoBackscatterProcessing
Some code I have written for a research project to automate some of the useful image processing techniques when analysing SAR backscatter changes





firstdraft.py takes an array of SAR files, read from the current directory, which have already had the relevant corrections, and reads the date they were taken. It then plots the backscatter in dB and the change between subsequent images in dB, and it also gives the option of taking the average of the previous n images, where n can be specified. I have chosen 1, but its useful to try other numbers to average over. 

LinearInversion.py takes implements a linear inversion on a specified number of images. In the file you can see the initial array of event dates that are specified, and then the number of days before and after each eruption/event to apply the inversion to. Some of the linear algebra useful to understand this transformation can be found: https://wiki.seg.org/wiki/Linear_inversion.

The essential transformation to understand is as follows:

$$\begin{bmatrix} d_1 \\\ d_2 \\\ d_3 \\\ d_4 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\\ 0 & 1 \\\ 0 & 1 \\\ 1 & 1 \end{bmatrix}  *  \begin{bmatrix} step \\\ prestep \end{bmatrix} = \begin{bmatrix} prestep \\\ prestep \\\ prestep \\\ step + prestep\end{bmatrix}$$

So the problem is essentially mapped into 2 dimensions. 
