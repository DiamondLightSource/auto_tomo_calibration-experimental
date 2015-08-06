import create_projections as projections
import detector as detect
import sort_watershed as sort
import find_resolution as resolution
import pylab as pl
import pickle

################## PARAMETERS #########################

R1 = 0.2
R2 = 0.2
C1 = (0., 0., 0.)
C2 = (0., -0.4, 0.)
size = 256
sampling = 360
name = "./data/analytical%i.tif"
results = "./results/result%i.txt"

############## GENERATE A SPHERE ######################

# projections.analytical_3D(R1, C1, 1., R2, C2, 1.5, size, sampling, name)

############### DETECT CIRCLES #########################

detect.detect(size, name, results)

############### SORT CENTRES ###########################

sort.analyse(size, results)

############### FIND RESOLUTION ########################

f = open("./sorted/centres.npy", 'r')
centroids = pickle.load(f)
f.close()

f = open("./sorted/radii.npy", 'r')
radius = pickle.load(f)
f.close()

print centroids
print radius
touch_c, touch_pt, radii = resolution.find_contact_3D(centroids, radius, tol = 3.)

# define crop size
sample = 1

for i in range(len(touch_c)):
    c1 = touch_c[i][0]
    c2 = touch_c[i][1]
    r1 = radii[i][0]
    r2 = radii[i][1]
    crop_size = max(int(r1),int(r2)) - 1
    
    crop = resolution.crop_area(c1, c2, r1, crop_size, name)
#     crop = gaussian_filter(crop, 1)
    print crop.shape
    
#     for slice in range(crop_size*2):
#         pl.imshow(crop[:,:, slice])
#         pl.gray()
#         pl.pause(0.05)
    
    pl.imshow(crop[:,:, crop_size])
    pl.gray()
    pl.show()
    
    mod_c1, mod_c2 = resolution.centres_shifted_to_box(c1, c2, crop_size)
    
    print mod_c1, mod_c2
    
    resolution.touch_lines_3D(mod_c1, mod_c2, crop, sample)
