import numpy as np
import pylab as pl
from skimage import io
from scipy.ndimage.filters import sobel
from scipy.ndimage.measurements import watershed_ift
from skimage.filter import denoise_tv_chambolle
import numpy as np
import pylab as pl
from scipy import ndimage, misc

from skimage import measure
from skimage import exposure
from skimage.morphology import watershed
from skimage.filter import threshold_otsu, sobel, denoise_tv_chambolle
from skimage.exposure import rescale_intensity
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter

def distance_3D(c1, c2):
    """
    Calculate the distance between two points
    """
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)

binary = np.load("/dls/tmp/jjl36382/50880/spheres/sphere2distance.npy")

local_maxi = peak_local_max(binary,# num_peaks=1,
                        indices=False)

# Markers correspond to the maxima inside the EDM
markers = ndimage.label(local_maxi)[0]

X = []
Y = []
Z = []

leng = markers.shape[2]
from sphere_fit import leastsq_sphere

for z in range(0, leng,10):
#     
#     coords = np.column_stack(np.nonzero(markers[:,:,z]))
#     Xtemp = np.array(coords[:, 0])
#     Ytemp = np.array(coords[:, 1])
#                     
#     X.extend(Xtemp)
#     Y.extend(Ytemp)
#     Z = np.hstack([np.repeat([z], len(Xtemp)), Z])
#     
    pl.imshow(markers[:,:,z])
    pl.gray()
    pl.pause(0.001)

p1, rsq = leastsq_sphere(X, Y, Z, leng / 2., (ndimage.center_of_mass(binary)[0], ndimage.center_of_mass(binary)[1]))

xc, yc, zc = ndimage.center_of_mass(binary)    
# Initial centre estimate
R = int(307.7*1.1)
initx, inity, initz = ((2*R + 1) / 2., (2*R + 1) / 2., (2*R + 1) / 2.)
 
# Correction of the centroids based on the initial estimate
corrx, corry, corrz = (initx - xc, inity - yc, initz - zc)
print "original", corrx, corry, corrz
print "best fit", initx - p1[0], inity - p1[1], initz - p1[2]
