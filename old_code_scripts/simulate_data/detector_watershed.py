import numpy as np
import pylab as pl

from skimage import io
from skimage import measure
from scipy import ndimage, misc
from skimage.morphology import watershed, disk
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu, rank
from scipy.ndimage.morphology import binary_opening, binary_closing

def preprocessing(image, smooth_size, folder):
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
    from skimage.restoration import denoise_tv_chambolle
    
    dim = int(image.shape[0] / 50.)
    smoothed = rank.median(image, disk(smooth_size))
    #smoothed = denoise_tv_chambolle(image, weight=0.002)
    smoothed = rank.enhance_contrast(smoothed, disk(smooth_size))
    
    pl.subplot(2, 3, 1)
    pl.title("after median")
    pl.imshow(smoothed)
    pl.gray()
    # If after smoothing the "dot" disappears
    # use the image value
    
    # TODO: wat do with thresh?
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
        bin_open = binary_opening(binary, np.ones((dim, dim)), iterations=5)
        bin_close = binary_closing(bin_open, np.ones((5,5)), iterations=5)
        
        pl.subplot(2, 3, 2)
        pl.title("threshold")
        pl.imshow(binary, interpolation='nearest')
        pl.subplot(2, 3, 3)
        pl.title("opening")
        pl.imshow(bin_open, interpolation='nearest')
        pl.subplot(2, 3, 4)
        pl.title("closing")
        pl.imshow(bin_close, interpolation='nearest')
        
        distance = ndimage.distance_transform_edt(bin_open)
        local_maxi = peak_local_max(distance,
                                    indices=False, labels=bin_open)
        
        markers = ndimage.label(local_maxi)[0]
        
        labeled = watershed(-distance, markers, mask=bin_open)
        pl.subplot(2, 3, 5)
        pl.title("label")
        pl.imshow(labeled)
        #pl.show()
        pl.savefig(folder)
        pl.close('all')

        #misc.imsave(folder, labeled)
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
    
    for info in measure.regionprops(image, ['Centroid', 'BoundingBox', 'Area', 'equivalent_diameter', 'Label']): 
        
        # Skip wrong regions
        index = np.where(image==info['Label'])
        if index[0].size==0 & index[1].size==0:
            continue
        
        # TODO: change this value
        if info['Area'] > image.shape[0] / 4.:
            
        
            centre = info['Centroid']
            D = info['equivalent_diameter']
            
            #min_row, min_col, max_row, max_col = info['BoundingBox']
            #a1 = int((max_row - min_row) / 2.)
            #a2 = int((max_col - min_col) / 2.)
            
            #box_cent = (a1 + min_row, a2 + min_col)
            
            radius.append(round(D / 2.0, 3))
            centroids.append( (round(centre[0], 3),round(centre[1], 3)) )
            #bords.append(box_cent)

    return [centroids, radius]


def add_noise(np_image, amount):
    """
    Adds random noise to the image
    """
    noise = np.random.randn(np_image.shape[0],np_image.shape[1])
    norm_noise = abs(noise/np.max(noise))
    np_image = np_image + norm_noise*np.max(np_image)*amount
    
    return np_image
# 
# 
# img = io.imread("./shifted_data/sino_00100.tif")
# # img = io.imread("test_slice.tif")
# pl.subplot(2, 3, 6)
# pl.title("original")
# pl.imshow(img)
# pl.gray()
#   
#         
# a, b, c = watershed_segmentation(img, 4, "ayy")
#           
# print a, b, c
