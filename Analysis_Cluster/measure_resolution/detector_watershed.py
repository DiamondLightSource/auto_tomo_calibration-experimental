import numpy as np

from skimage import io
from skimage import measure
from scipy import ndimage, misc
from skimage.morphology import watershed, disk
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu, rank, denoise_tv_chambolle
from scipy.ndimage.morphology import binary_opening, binary_closing, binary_fill_holes, binary_erosion, binary_dilation
from scipy.ndimage.filters import median_filter, gaussian_filter


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
    smoothed = gaussian_filter(image, 5)
#     misc.imsave(folder + 'smooth%05i.jpg' % task_id, smoothed)
    
    # TODO: what do with thresh?
    try:
        im_max = smoothed.max()
        thresh = threshold_otsu(smoothed)
    except:
        im_max = image.max()
        thresh = threshold_otsu(image)

    
    if im_max < thresh:
        labeled = np.zeros(smoothed.shape, dtype=np.int32)
        
    else:
        binary = smoothed > thresh
     
        # Open the image by connecting small cracks and remove salt
        bin_open = binary_opening(binary, np.ones((5, 5)), iterations=5)

        # Close up any small cracks
        bin_close = binary_closing(bin_open, np.ones((5, 5)), iterations=5)
        
        # Fill the holes inside the circles
        bin_fill = binary_fill_holes(bin_close)
        
        
#         # Dilated image to get the regions we are sure aren't circles
#         background = binary_dilation(bin_fill, np.ones((5, 5)), iterations=5)
#          
#         # Foreground
#         dist_transform = ndimage.distance_transform_edt(bin_fill)
#         foreground = (dist_transform > threshold_otsu(dist_transform))
#          
#         # Unknown region
#         unknown = (background - foreground) * 1
#          
#         # Marker labelling
#         marks = ndimage.label(foreground)[0]
#          
#         # Add one to all labels so that sure background is not 0, but 1
#         marks = marks+1
#          
#         # Now, mark the region of unknown with zero
#         #print unknown
#         marks[unknown==1.] = 0
#  
#         # water
#         water = watershed(image, marks)
        
        distance = ndimage.distance_transform_edt(bin_fill)
        local_maxi = peak_local_max(distance,
                                    indices=False, labels=bin_fill)
        
        markers = ndimage.label(local_maxi)[0]
        
        labeled = watershed(-distance, markers, mask=bin_fill)
        
#         pl.subplot(2, 3, 1)
#         pl.title("filtered")
#         pl.imshow(smoothed)
#         pl.gray()
#         pl.subplot(2, 3, 2)
#         pl.title("opened")
#         pl.imshow(bin_open)
#         pl.subplot(2, 3, 3)
#         pl.title("closed")
#         pl.imshow(bin_close)
#         pl.subplot(2, 3, 4)
#         pl.title("filled")
#         pl.imshow(bin_fill)
#         pl.subplot(2, 3, 5)
#         pl.title("label")
#         pl.imshow(water)
#         pl.show()
#         pl.savefig(folder)
#         pl.close('all')
        misc.imsave(folder + 'labels%05i.jpg' % task_id, labeled)

#         labels_rw = random_walker(bin_close, markers, mode='cg_mg')
#         pl.imshow(labels_rw, interpolation='nearest')
#         pl.show()

    return labeled


def watershed_segmentation(image, smooth_size, folder, task_id):
    
    if np.unique(image)[0] == 0.:
        return [[], []]
    
    labels = preprocessing(image, smooth_size, folder, task_id)
    
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
 
# from skimage import io
# import pylab as pl
#   
# img = io.imread("/dls/science/groups/das/ExampleData/SphereTestData/38644/recon_01400.tif")
# 
# watershed_segmentation(img, 3, 1, 1)
