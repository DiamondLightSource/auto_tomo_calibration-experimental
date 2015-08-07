import create_projections as projections
import detector as detect
import sort_watershed as sort
import find_resolution as resolution
import pylab as pl
import pickle
import os


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

################## PARAMETERS #########################

R1 = 0.3
R2 = 0.3
C1 = (0., 0., 0.)
C2 = (0., -0.6, 0.)
size = 300
sampling = 180
median = 5

# just change then ends of the folders for different data sets
name = "./data_highdif/analytical%i.tif"
results = "./results_highdif/result%i.txt"
sorted = "./sorted_highdif/"
plots = "./plots_highdif/"

create_dir("./data_highdif/")
create_dir("./results_highdif/")

############## GENERATE A SPHERE ######################

#projections.analytical_3D(R1, C1, 1., R2, C2, 5., size, sampling, name)

############### DETECT CIRCLES #########################

#detect.detect(size, name, results, median)

############### SORT CENTRES ###########################

#sort.analyse(size, results, sorted)

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
sample = 1

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
    
    print "shape", crop.shape
    
    resolution.touch_lines_3D(mod_c1, mod_c2, crop, sample, crop_size, plots)
