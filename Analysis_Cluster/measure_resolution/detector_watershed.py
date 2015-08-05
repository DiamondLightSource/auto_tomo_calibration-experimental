import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage import measurements
from scipy import optimize

from skimage import io
from skimage import measure, color
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu, sobel
from skimage.filter import denoise_tv_chambolle
from skimage.util import img_as_ubyte
from scipy.ndimage.filters import median_filter, gaussian_filter


def save_data(filename, data):
    import pickle
    print("Saving data")
    f = open(filename, 'w')
    pickle.dump(data, f)
    f.close()

  
def watershed_segmentation(image):
    
    image = denoise_tv_chambolle(image, weight=0.002)
    
    nbins = 50
    thresh = threshold_otsu(image, nbins)
    image = (image > thresh) * 1
     
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
    
    centroids, areas, bord, radius = centres_of_mass_2D(labels)
    
    print "before", centroids

    # Check the areas
    index = []
    size = max(image.shape[0] / 2, image.shape[1] / 2)

    for i in range(0, len(areas)):
        # Jump too big or too small areas
        if areas[i].shape[0] >= size or areas[i].shape[1] >= size\
        or areas[i].shape[0] <= size/5 or areas[i].shape[1] <= size/5:
            index.append(i)
            continue
    
    if index != []:
        areas[:] = [item for i,item in enumerate(areas) if i not in index]
        bord[:] = [item for i,item in enumerate(bord) if i not in index]
    
    index = []
    for i in range(1, len(areas)):
        # Jump almost same areas (= same circle)
        if (abs(bord[i][0] - bord[i-1][0]) <= 100 and abs(bord[i][2] - bord[i-1][2]) <= 100):
            index.append(i)
            continue
    
    if index != []:
        centroids[:] = [item for i,item in enumerate(centroids) if i not in index]
        radius[:] = [item for i,item in enumerate(radius) if i not in index]
        bord[:] = [item for i,item in enumerate(bord) if i not in index]
        areas[:] = [item for i,item in enumerate(areas) if i not in index]

    from skimage.draw import circle_perimeter
    
    circles = []
    
    for cent in range(len(bord)):
        minr, minc, maxr, maxc = bord[cent]
        xc = (maxr - minr) / 2.
        yc = (maxc - minc) / 2.
        cx, cy = circle_perimeter(int(xc), int(yc), int(radius[cent]))
        circles.append((cy, cx))
        
    print "after", centroids
          
    return [centroids, areas, bord, radius, circles]



def centres_of_mass_2D(image):
    """
    Calculates centres of mass
    for all the labels
    """
    centroids = []
    bords = []
    areas = []
    radius = []
    
    for info in measure.regionprops(image, ['Centroid', 'BoundingBox', 'Area', 'equivalent_diameter']): 
        
        # Skip small regions
        if info['Area'] > 100:
            centre = info['Centroid']
            minr, minc, maxr, maxc = info['BoundingBox']
            D = info['equivalent_diameter']
            
            margin = 0
            
            radius.append((D / 2.0))
            bords.append((minr-margin, minc-margin, maxr+margin, maxc+margin))
            areas.append(image[minr-margin:maxr+margin,minc-margin:maxc+margin].copy())
            centroids.append(centre)
            
        return [centroids, areas, bords, radius]


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
        
    return [bords, slice_centroids, slice_radius]

# img = io.imread("/dls/tmp/tomas_aidukas/scans_july16/cropped/50867/image_01500.tif")
# pl.imshow(img)
# pl.show()
#  
# a, b, c, d, e = watershed_segmentation(img)
# 
# print e

