import create_projections as projections
import pylab as pl
import pickle
import os
import numpy as np
"""
Sum over slices of segmented images and see which
slice has the highest sum across the whole image.
The one with the highest number is the centre image
"""

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

################## PARAMETERS #########################

# dim/2 * to get true values
R1 = 0.3
R2 = 0.3
C1 = (0.3, 0.3, 0.)
#X = 0.3
#Y = round(np.sqrt(((R1 + R2) * np.cos(np.radians(45))) ** 2 - X ** 2), 2)
#C2 = ( X, Y,round((R1 + R2) * np.sin(np.radians(45)), 2)) # 45 degree angle contact
C2 = (-0.3, 0.3, 0.)
x,y,z = C2
size = 300 #total image dimensions
sampling = 360
median = 3

# just change then ends of the folders for different data sets
folder_start = "./contrast/"
name = folder_start + "data/sino_%05i.tif"
label_name = folder_start + "label/analytical%i.png"
results = folder_start + "results/result%i.txt"
sorted = folder_start + "sorted/"
plots = folder_start + "plots/"

create_dir(folder_start + "data/")
create_dir(folder_start + "results/")
create_dir(folder_start + "label/")

############## GENERATE A SPHERE ######################

#projections.analytical_3D(R1, C1, 1., R2, C2, 0.5, size, sampling, name)
sphere = projections.sphere(R1, R2, C1, C2, 0.5, 0.5, size)
projections.get_projections_3D(sphere, size, name, sampling)

############### DETECT CIRCLES #########################

#detect.detect(size, name, results, median, label_name)

############### SORT CENTRES ###########################

#sort.analyse(size, results, sorted)

############### FIND RESOLUTION ########################

