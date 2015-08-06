import numpy as np
import pylab as pl

from skimage import io
from scipy import ndimage as ndi
from skimage import measure
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu
from skimage.segmentation import random_walker


def preprocessing(image, smooth_size, min_r):
    """
    'The image low contrast and under segmentation
    problem is not yet addressed by most of the researchers'
    
    'Other researchers also proposed different method to
    remedy the problem of watershed.  Li, El-
    moataz, Fadili, and Ruan, S. (2003) proposed an improved
    image segmentation approach based 
    on level set and mathematical morphology' 
    """
    from skimage.filter import threshold_otsu, threshold_adaptive, rank
    from skimage.morphology import label
    from skimage.measure import regionprops
    from skimage.feature import peak_local_max
    from scipy import ndimage
    from skimage.morphology import disk, watershed
    from scipy.ndimage.morphology import binary_opening, binary_closing
    
    
    thresh = threshold_otsu(image)
    
    smoothed = rank.median(image, disk(smooth_size))
    smoothed = rank.enhance_contrast(smoothed, disk(smooth_size))
    
    im_max = smoothed.max()
    
    if im_max < thresh:
        labeled = np.zeros(smoothed.shape, dtype=np.int32)
    else:
        binary = smoothed > thresh
        bin_open = binary_opening(binary)
        bin_close = binary_opening(bin_open)
        binary = bin_close
        pl.imshow(binary, interpolation='nearest')
        pl.show()
        pl.imshow(bin_open, interpolation='nearest')
        pl.show()
        pl.imshow(bin_close, interpolation='nearest')
        pl.show()
        
        distance = ndimage.distance_transform_edt(bin_close)
        local_maxi = peak_local_max(distance,
                                    indices=False, labels=bin_close)
        
        markers = ndimage.label(local_maxi)[0]
        
        labeled = watershed(-distance, markers, mask=bin_close)
#         labels_rw = random_walker(bin_close, markers, mode='cg_mg')
#         
#         pl.imshow(labels_rw, interpolation='nearest')
#         pl.show()
        
    pl.imshow(labeled, interpolation='nearest')
    pl.show()
    
    return labeled

def watershed_segmentation(image, do, smooth_size, min_r):
    
    labels = preprocessing(image, smooth_size, min_r)
    
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
    
    for info in measure.regionprops(image, ['Centroid', 'BoundingBox', 'Area', 'equivalent_diameter', 'Label']): 
        
        # Skip wrong regions
        index = np.where(image==info['Label'])
        if index[0].size==0 & index[1].size==0:
            continue
        
        # TODO: change this value
        if info['Area'] > image.shape[0] / 4.:
            
        
            centre = info['Centroid']
            D = info['equivalent_diameter']
            
            radius.append(round(D / 2.0, 3))
            centroids.append(
                             (round(centre[0], 3),round(centre[1], 3))
                             )

    return [centroids, radius]


def add_noise(np_image, amount):
    """
    Adds random noise to the image
    """
    noise = np.random.randn(np_image.shape[0],np_image.shape[1])
    norm_noise = abs(noise/np.max(noise))
    np_image = np_image + norm_noise*np.max(np_image)*amount
    
    return np_image


#img = io.imread("./data/analytical128.tif")
img = io.imread("test_slice.tif")
#img = add_noise(img, 0.5)
pl.imshow(img)
pl.gray()
pl.show()

a, b = watershed_segmentation(img, 0, 5, 4)
  
print a, b
