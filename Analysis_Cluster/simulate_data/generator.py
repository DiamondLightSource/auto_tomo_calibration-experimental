import create_projections as projections
import detector as detect
import sort_watershed as sort
import find_resolution as resolution
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
C1 = (0., 0., 0.)
X = 0.3
Y = round(np.sqrt((R1 + R2) * np.cos(np.radians(45)) ** 2 - X ** 2),2)
C2 = ( X, Y,round((R1 + R2) * np.sin(np.radians(45)), 2)) # 45 degree angle contact
x,y,z = C2
size = 200 # total image dimensions
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

projections.analytical_3D(R1, C1, 1., R2, C2, 0.5, size, sampling, name)
#sphere = projections.sphere(R1, R2, C1, C2, 0.5, 0.5, size)
#projections.get_projections_3D(sphere, size, name, sampling)

############### DETECT CIRCLES #########################

detect.detect(size, name, results, median, label_name)

############### SORT CENTRES ###########################

sort.analyse(size, results, sorted)

############### FIND RESOLUTION ########################

f = open(sorted + "centres.npy", 'r')
centroids = pickle.load(f)
f.close()
f = open(sorted +"radii.npy", 'r')
radius = pickle.load(f)
f.close()

print "centres of spheres", centroids
print "radii of spheres", radius
touch_c, touch_pt, radii = resolution.find_contact_3D(centroids, radius, tol = 3.)

# define sampling size
sample = 2
    
for i in range(len(touch_c)):
    c1 = touch_c[i][0]
    c2 = touch_c[i][1]
    r1 = radii[i][0]
    r2 = radii[i][1]
    
    resolution.touch_lines_3D(c1, c2, sample, plots,name)
