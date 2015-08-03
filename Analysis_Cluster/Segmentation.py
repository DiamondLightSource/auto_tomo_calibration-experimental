import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage import measurements
from scipy import optimize
import EqnLine as line

from skimage import io
from skimage import measure, color
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu, sobel
from skimage.filter import denoise_tv_chambolle
from skimage.util import img_as_ubyte
from scipy.ndimage.filters import median_filter, gaussian_filter
from peak_detect import *

import pickle


def save_data(filename, data):
    import pickle
    print("Saving data")
    f = open(filename, 'w')
    pickle.dump(data, f)
    f.close()


def crop_box_3D(image, touch_pt, centres, size = 30):
    """
    Crop a region around the touch point
    and perform Siemens star resolution
    analysis
    """
    crops = []
    for i in range(len(touch_pt)):
        
        c1 = centres[i][0]
        c2 = centres[i][1]

        crop = image[int(touch_pt[i][0]) - size:int(touch_pt[i][0]) + size,
                     int(touch_pt[i][1]) - size:int(touch_pt[i][1]) + size,
                     int(touch_pt[i][2]) - size:int(touch_pt[i][2]) + size]
        
#         pl.imshow(crop[:,:,30])
#         pl.gray()
#         pl.show()
        crops.append(crop)
        

    return crops

def watershed_3d(sphere):
    """
    Markers should be int8
    Image should be uint8
    """
   
    sphere = median_filter(sphere, 3)
    thresh = threshold_otsu(sphere)
    sphere = (sphere >= thresh) * 1
    sphere = sobel(sphere)
    
    size = (sphere.shape[0], sphere.shape[1], sphere.shape[2])
    
    marker = np.zeros(size, dtype=np.int16)
    pl.imshow(sphere[:,:,50])
    pl.show()
    # mark everything outside as background
    marker[5, :, :] = -1
    marker[size[0] - 5, :, :] = -1
    marker[:, :, 5] = -1
    marker[:, :, size[2] - 5] = -1
    marker[:, 5, :] = -1
    marker[:, size[1] - 5, :] = -1
    marker[:,0,0] = -1
    # mark everything inside as a sphere
    marker[size[0] / 2., size[1] / 2., size[2] / 2.] = 5

    result = measurements.watershed_ift(sphere.astype(dtype=np.uint16), marker)
    pl.imshow(result[:,:,50])
    pl.show()
    
    return result

  
def watershed_segmentation(image):

    
#     #threshold
#     image = median_filter(image, 5)
#  
#     filter = threshold_otsu(image)
#     image = (image > filter) * 1
     
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image)
    
#     fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7))
#     ax0, ax1, ax2 = axes
#     
#     ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
#     ax0.set_title('Overlapping objects')
#     ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
#     ax1.set_title('Distances')
#     ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
#     ax2.set_title('Separated objects')
#     
#     for ax in axes:
#         ax.axis('off')
#     
#     fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
#                         right=1)
#     plt.show()
    
    

    
    return labels


def centres_of_mass_2D(image):
    """
    Calculates centres of mass
    for all the labels
    """
    centroids = []
    bords = []
    areas = []
    radius = []

    for info in measure.regionprops(image, ['Centroid', 'BoundingBox', 'equivalent_diameter']): 
        
        centre = info['Centroid']
        minr, minc, maxr, maxc = info['BoundingBox']
        D = info['equivalent_diameter']
    
        
        margin = 0
        
        radius.append((D / 2.0))
        bords.append((minr-margin, minc-margin, maxr+margin, maxc+margin))
        areas.append(image[minr-margin:maxr+margin,minc-margin:maxc+margin].copy())
        centroids.append(centre)
        
    return centroids, areas, bords, radius


def watershed_slicing(image):
    """
    Does the watershed algorithm slice by slice.
    Then use the labeled image to calculate the centres of
    mass for each slice.
    """
    image = median_filter(image, 3)
    thresh = threshold_otsu(image)
    image = (image > thresh) * 1
    
    
    N = len(image)
    slice_centroids = []
    slice_radius = []
    
    for i in range(N):
        
        slice = image[:, :, i]
        
        labels_slice = watershed_segmentation(slice)
        centroids, areas, bords, radius = centres_of_mass_2D(labels_slice)
        
        slice_centroids.append(centroids)
        slice_radius.append(radius)
#         if i > 49:
#             print centroids
#             pl.imshow(labels_slice)
#             pl.show()
        
    return slice_centroids, slice_radius

################# DRAW TEST DATA ######################################

def draw_sphere():
    
    import numpy as np
    
    sphere = np.zeros((100, 100 ,100))
    N = 100
    radius1 = 20
    radius2 = 20
    centre1 = (30, 30, 50)
    centre2 = (30, 69, 50)

    Xc1 = centre1[0]
    Yc1 = centre1[1]
    Zc1 = centre1[2]
    
    Xc2 = centre2[0]
    Yc2 = centre2[1]
    Zc2 = centre2[2]
    
    Y, X, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
    mask1 = (((X - Xc1)**2 + (Y - Yc1)**2 + (Z - Zc1)**2) < radius1**2)
    mask2 = (((X - Xc2)**2 + (Y - Yc2)**2 + (Z - Zc2)**2) < radius2**2)
    sphere[mask1] = 1
    sphere[mask2] = 1
    
    return sphere


def add_noise(np_image, amount):
    import numpy as np
    noise = np.random.randn(np_image.shape[0],np_image.shape[1],np_image.shape[2])
    norm_noise = noise/np.max(noise)
    np_image = np_image + norm_noise*np.max(np_image)*amount
    return np_image

#############################################################################


from test_analysis import test_analyse

sphere = draw_sphere()
sphere = add_noise(sphere, 0.3)
#sphere = gaussian_filter(sphere, 3)

centroids, radii = watershed_slicing(sphere)
rad, cent = test_analyse.analyse(radii, centroids)
touch_pt, centres = line.find_contact_3D(cent, rad, tol = 2)

#crop_img, slope = crop_box_3D(sphere, touch_pt, centres, size = 30)

pt1 = cent[0]
pt2 = cent[1]

line.touch_lines_3D(pt1, pt2, sphere)


# image = io.imread("test_slice.tif")
# sphere = np.load('sphere1.npy')
# centroids, radii = watershed_slicing(sphere)
# save_data("test_analysis/centroids.dat", centroids)
# save_data("test_analysis/radii.dat", radii)
# labels = watershed_segmentation(image)

#
# centroids, areas, bords, radius, radius2 = centres_of_mass_2D(labels)
# 
# # leastsq_circle_fit(areas, centroids, bords, radius)
# # leastsq_whole(image, centroids)
# touch, centres = find_contact(centroids, radius2)
# 
# crop_img, slopes = crop_box(image, touch, centres)
# 
# line.eqn_line(crop_img[0], slopes[0]) 