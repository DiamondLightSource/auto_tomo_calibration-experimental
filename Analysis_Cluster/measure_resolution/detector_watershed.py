import numpy as np
import pylab as pl

from skimage import io
from skimage import measure
from scipy import ndimage, misc
from skimage.morphology import watershed, disk
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu, rank
from scipy.ndimage.morphology import binary_opening, binary_closing


def preprocessing(image, smooth_size, folder, task_id):
    """
    'The image low contrast and under segmentation
    problem is not yet addressed by most of the researchers'
    
    'Other researchers also proposed different method to
    remedy the problem of watershed.  Li, El-
    moataz, Fadili, and Ruan, S. (2003) proposed an improved
    image segmentation approach based 
    on level set and mathematical morphology'
    
    THE SPHERES MUST BE ALMOST ALONG THE SAME PLANES IN Z DIRECTION
    IF THEY ARE TOUCHING AND OVERLAP, WHILE BEING ALMOST MERGED
    IT IS IMPOSSIBLE TO RESOLVE THEM
    
    ONE IDEA MIGHT BE TO DETECT CENTRES ALONG ONE AXIS AND THEN ANOTHER
    AFTER ALL THE CENTRES WERE FOUND COMBINE THEM SOMEHOW... 
    """
    
    smoothed = rank.median(image, disk(smooth_size))
    smoothed = rank.enhance_contrast(smoothed, disk(smooth_size))
    
#     pl.subplot(2, 3, 1)
#     pl.title("after median")
#     pl.imshow(smoothed)
#     pl.gray()
    # If after smoothing the "dot" disappears
    # use the image value
    
    # TODO: what do with thresh?
    try:
        im_max = smoothed.max()
        thresh = threshold_otsu(image)
    except:
        im_max = image.max()
        thresh = threshold_otsu(image)

    
    if im_max < thresh:
        labeled = np.zeros(smoothed.shape, dtype=np.int32)
        
    else:
        binary = smoothed > thresh
        
        # TODO: this array size is the fault of errors
        bin_open = binary_opening(binary, np.ones((5, 5)), iterations=5)
        
#         pl.subplot(2, 3, 2)
#         pl.title("threshold")
#         pl.imshow(binary, interpolation='nearest')
#         pl.subplot(2, 3, 3)
#         pl.title("opening")
#         pl.imshow(bin_open, interpolation='nearest')
#         pl.subplot(2, 3, 4)
#         pl.title("closing")
#         pl.imshow(bin_close, interpolation='nearest')
        
        distance = ndimage.distance_transform_edt(bin_open)
        local_maxi = peak_local_max(distance,
                                    indices=False, labels=bin_open)
        
        markers = ndimage.label(local_maxi)[0]
        
        labeled = watershed(-distance, markers, mask=bin_open)
#         pl.subplot(2, 3, 5)
#         pl.title("label")
#         pl.imshow(labeled)
#         #pl.show()
#         pl.savefig(folder)
#         pl.close('all')
        
        misc.imsave(folder + 'labels%05i.jpg' % task_id, labeled)
#         labels_rw = random_walker(bin_close, markers, mode='cg_mg')
#          
#         pl.imshow(labels_rw, interpolation='nearest')
#         pl.show()

    return labeled


def watershed_segmentation(image, smooth_size, folder):
    
    if np.unique(image)[0] == 0.:
        return [[], []]
    
    labels = preprocessing(image, smooth_size, folder)
    
    centroids, radius = centres_of_mass_2D(labels)
    
    print centroids
    
    return [centroids, radius]



def centres_of_mass_2D(image):
    """
    Calculates centres of mass
    for all the labels
    """
    centroids = []
    bords = []
    areas = []
    radius = []
    
    for info in measure.regionprops(image, ['Centroid', 'Area', 'equivalent_diameter', 'Label']): 
        
        # Skip wrong regions
        index = np.where(image==info['Label'])
        if index[0].size==0 & index[1].size==0:
            continue
        
        # TODO: change this value
        if info['Area'] > image.shape[0] / 4.:
            
        
            centre = info['Centroid']
            D = info['equivalent_diameter']

            
            radius.append((D / 2.0))
            centroids.append(centre)

    return [centroids, radius]