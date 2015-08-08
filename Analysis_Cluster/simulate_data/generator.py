import create_projections as projections
import detector as detect
import sort_watershed as sort
import find_resolution as resolution
import pylab as pl
import pickle
import os
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
R1 = 0.4
R2 = 0.4
C1 = (0., 0., 0.)
C2 = (0., -0.4, 0.4)
size = 100 # total image dimensions
sampling = 360
median = 3

# just change then ends of the folders for different data sets
folder_start = "./shifted_"
name = folder_start + "data/analytical%i.tif"
label_name = folder_start + "label/analytical%i.png"
results = folder_start + "results/result%i.txt"
sorted = folder_start + "sorted/"
plots = folder_start + "plots/"

create_dir(folder_start + "data/")
create_dir(folder_start + "results/")
create_dir(folder_start + "label/")

############## GENERATE A SPHERE ######################

projections.analytical_3D(R1, C1, 1., R2, C2, 1., size, sampling, name)

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
    
    # Cropsize is the size away from the point of contact
    # in one and another direction
    crop_size = max(int(r1),int(r2))
    crop_size = int(crop_size / 2.)    
    
    crop = resolution.crop_area(c1, c2, r1, crop_size, name)
#     crop = gaussian_filter(crop, 1)

    mod_c1, mod_c2 = resolution.centres_shifted_to_box(c1, c2, crop_size)
    
    resolution.touch_lines_3D(mod_c1, mod_c2, crop, sample, crop_size, plots)
