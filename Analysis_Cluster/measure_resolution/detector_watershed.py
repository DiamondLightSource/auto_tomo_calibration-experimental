import numpy as np
import pylab as pl

from skimage import measure, io
from scipy import ndimage, misc, optimize
from skimage.morphology import watershed
from skimage.filter import threshold_otsu, sobel
from scipy.ndimage.morphology import binary_opening, binary_closing, binary_fill_holes, binary_erosion, binary_dilation
from scipy.ndimage.filters import gaussian_filter, median_filter
from skimage.transform import hough_circle
from skimage.feature import peak_local_max


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
    smoothed = median_filter(image, 3)
#     smoothed = gaussian_filter(smoothed, 3)
#     misc.imsave(folder + 'smooth%05i.jpg' % task_id, smoothed)
    
    # TODO: what do with thresh?
    im_max = smoothed.max()
    thresh = threshold_otsu(smoothed)
    
    binary = smoothed > thresh
    
    # Open the image by connecting small cracks and remove salt
    bin_open = binary_opening(binary, np.ones((5, 5)), iterations=5)

    # Close up any small cracks
    bin_close = binary_closing(bin_open, np.ones((5, 5)), iterations=10)

    # Fill the holes inside the circles
    #bin_fill = binary_fill_holes(bin_close, np.ones((5, 5)))
    
    distance = ndimage.distance_transform_edt(bin_close)
    local_maxi = peak_local_max(distance,
                                indices=False, labels=bin_close)
    
    markers = ndimage.label(local_maxi)[0]
    
    labeled = watershed(-distance, markers, mask=bin_close)
    
#         pl.subplot(2, 3, 1)
#         pl.title("binary")
#         pl.imshow(binary)
#         pl.gray()
#         pl.subplot(2, 3, 2)
#         pl.title("opened")
#         pl.imshow(bin_open)
#         pl.subplot(2, 3, 3)
#         pl.title("closed")
#         pl.imshow(bin_close)
#         pl.subplot(2, 3, 4)
# #         pl.title("filled")
# #         pl.imshow(bin_fill)
#         pl.subplot(2, 3, 5)
#         pl.title("local_maxi")
#         pl.imshow(distance)
#         pl.subplot(2, 3, 6)
#         pl.title("label")
#         pl.imshow(labeled)
#         pl.show()
#         pl.close('all')
    
#         if task_id % 10 == 0:
#             misc.imsave(folder + 'labels%05i.jpg' % task_id, labeled)

#         labels_rw = random_walker(bin_close, markers, mode='cg_mg')
#         pl.imshow(labels_rw, interpolation='nearest')
#         pl.show()

    return labeled


def watershed_segmentation(image, smooth_size, folder, task_id):
    
#     if np.unique(image)[0] == 0.:
#         return [[], []]
#     
    labels = preprocessing(image, smooth_size, folder, task_id)
    
    centroids, radius = centres_of_mass_2D(labels, folder, task_id)
    
    print centroids
    print radius
    
    return [centroids, radius]


def centres_of_mass_2D(image, folder, task_id):
    """
    Calculates centres of mass
    for all the labels
    """
    from circle_fit import leastsq_circle
    centroids_fit = []
    radius_fit = []
    
   
    for info in measure.regionprops(image, ['coords ', 'Label', 'Area', 'equivalent_diameter', 'Centroid']): 
    
        if info['Area'] > image.shape[0] / 4.:
            
            edges = sobel(np.pad(info.image, 30, mode="constant"))
            
            coords = np.column_stack(np.nonzero(edges))
            X = coords[:,0]
            Y = coords[:,1]
            
            
            XC, YC, RAD, RESID = leastsq_circle(X, Y)
            minr, minc, maxr, maxc = info.bbox
            
            if info.area * 1.5 > np.pi*RAD**2:
                centroids_fit.append((minr - 30 + XC, minc - 30 + YC))
                radius_fit.append(RAD)
            
    # Check the areas
    index = []
    print "before clearing number of circles was", len(centroids_fit)
    
    for i in range(len(centroids_fit)):
        C = centroids_fit[i]
        R = radius_fit[i]
        
        if C[0] < 0 or C[1] < 0 or R < 0 or R > image.shape[0] / 2.:
            index.append(i)
    
    if index != []:
        centroids_fit[:] = [item for i,item in enumerate(centroids_fit) if i not in index]
        radius_fit[:] = [item for i,item in enumerate(radius_fit) if i not in index]

    print "after removal, number of circles is", len(centroids_fit)
    
    # Plot stuff
    if centroids_fit:
        for cent in centroids_fit:
            xc, yc = cent
            image[xc, yc] = 0
            image[xc-5:xc+5, yc-5:yc+5] = 0
    
    misc.imsave(folder + 'labels%05i.jpg' % task_id, image)
#     pl.imshow(image)
#     pl.show()
    return [centroids_fit, radius_fit]

# img = io.imread("/dls/science/groups/das/ExampleData/SphereTestData/38644/recon_01195.tif")
#            
# watershed_segmentation(img, 3, 1, 1)
