import numpy as np
import pylab as pl

from skimage import io
from scipy import ndimage as ndi
from skimage import measure
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import random_walker
from skimage import img_as_ubyte, dtype_limits


def preprocessing(image):
    """
    'The image low contrast and under segmentation
    problem is not yet addressed by most of the researchers'
    
    'Other researchers also proposed different method to
    remedy the problem of watershed.  Li, El-
    moataz, Fadili, and Ruan, S. (2003) proposed an improved
    image segmentation approach based 
    on level set and mathematical morphology' 
    """
    
    # imhist is the frequencies and bins are the pixel vlaues
    imhist, bins = np.histogram(image.flatten(), 256)
    
    max_freq = np.argwhere(max(imhist) == imhist)[0][0]
    min_freq = np.argwhere(min(imhist) == imhist)
    print min(bins)
    
    # determine markers of the two phases from the
    # extreme tails of the histogram of gray values
    markers = np.zeros(image.shape, dtype=np.uint)
    markers[image < bins[min(min_freq.T[0])]] = 1
    markers[image > bins[max_freq]] = 2
    
    pl.imshow(markers)
    pl.gray()
    pl.show()
    
    
    # Run random walker algorithm
    labels = random_walker(image, markers, beta=10, mode='bf')
    #abels = image - labels
    pl.imshow(labels, interpolation='nearest')
    pl.show()
    
    return image

def watershed_segmentation(image, do):
    
#     image = denoise_tv_chambolle(image)
    image = ndi.filters.gaussian_filter(image, 3)
    if do == 1:
        pl.imshow(image)
        pl.gray()
        pl.show()
    
    thresh = threshold_otsu(image)
    image = (image > thresh) * 1
    
#     pl.imshow(image)
#     pl.show()
    
    distance = ndi.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),labels=image)
    markers = measure.label(local_maxi)
    labels = watershed(-distance, markers, mask=image)
    
    if do == 1:
        pl.imshow(labels)
        pl.gray()
        pl.show()
#     pl.imshow(labels)
#     pl.show()
    
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
            
            radius.append((D / 2.0))
            centroids.append(centre)

    return [centroids, radius]


def add_noise(np_image, amount):
    """
    Adds random noise to the image
    """
    noise = np.random.randn(np_image.shape[0],np_image.shape[1])
    norm_noise = abs(noise/np.max(noise))
    np_image = np_image + norm_noise*np.max(np_image)*amount
    
    return np_image


img = io.imread("./data/analytical128.tif")
img = add_noise(img, 0.3)
# pl.imshow(img)
# pl.gray()
# pl.show()

preprocessing(img)
a, b = watershed_segmentation(img)
  
print a, b
