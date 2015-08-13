import numpy as np

from skimage import io
from skimage import measure
from scipy import ndimage, misc
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu
from scipy.ndimage.morphology import binary_opening, binary_closing, binary_fill_holes, binary_erosion, binary_dilation
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import CircleModel, ransac


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
    #smoothed = median_filter(image, 3)
    smoothed = gaussian_filter(image, 7)
#     misc.imsave(folder + 'smooth%05i.jpg' % task_id, smoothed)
    
    # TODO: what do with thresh?
    try:
        im_max = smoothed.max()
        thresh = threshold_otsu(smoothed)
    except:
        im_max = image.max()
        thresh = threshold_otsu(smoothed)

    if im_max < thresh:
        labeled = np.zeros(smoothed.shape, dtype=np.int32)
        
    else:
        binary = smoothed > thresh
        
        # Open the image by connecting small cracks and remove salt
        bin_open = binary_opening(binary, np.ones((10, 10)), iterations=10)

        # Close up any small cracks
        bin_close = binary_closing(bin_open, np.ones((10, 10)), iterations=10)

        # Fill the holes inside the circles
        #bin_fill = binary_fill_holes(bin_close, np.ones((5, 5)))
      
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
#         pl.savefig(folder)
#         pl.close('all')
        if task_id % 10 == 0:
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
    centroids_fit = []
    radius_fit = []
    
    #for info in measure.regionprops(image, ['Centroid','BoundingBox', 'equivalent_diameter', 'Label', 'coords ']): 
    for info in measure.regionprops(image, ['coords ', 'Label', 'Area']): 
        
        # Skip wrong regions
#         index = np.where(image==info['Label'])
#         if index[0].size!=0 & index[1].size!=0:
        if info['Area'] > image.shape[0] / 4.:
            ransac_model, inliers = ransac(info['coords'], CircleModel, 3, 3, max_trials=100)
            params = ransac_model.params
        
            if params != []:
                radius_fit.append(params[2])
                centroids_fit.append((params[0], params[1]))
            else:
                print "Nothing was fitted!"
                    
    # Check the areas
    index = []
    print "before clearing number of circles was", len(centroids_fit)
    
    for i in range(len(centroids_fit)):
        C = centroids_fit[i]
        R = radius_fit[i]
        
        if C[0] < 0 or C[1] < 0 or R < 0 or R > image.shape[0]:
            index.append(i)
    
    if index != []:
        centroids_fit[:] = [item for i,item in enumerate(centroids_fit) if i not in index]
        radius_fit[:] = [item for i,item in enumerate(radius_fit) if i not in index]
    
    index = []
    N = len(centroids_fit)
    # The distance between centroids must be about equal to
    # the sum of their radii
    for i in range(N - 1):
        for j in range(i + 1, N):
            c1 = centroids_fit[i]
            c2 = centroids_fit[j]
            r1 = radius_fit[i]
            r2 = radius_fit[j]
            
            D = r1 + r2
            L = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
            # Distance between centres must be bigger than their radii sum
            # or about the same
            if D * 0.9 > L:
                index.append(i)
                index.append(j)
            continue
    
    if index != []:
        centroids_fit[:] = [item for i,item in enumerate(centroids_fit) if i not in index]
        radius_fit[:] = [item for i,item in enumerate(radius_fit) if i not in index]
    
    print "after removal number of circles is", len(centroids_fit)
    
    return [centroids_fit, radius_fit]
  
  
# from skimage import io
# import pylab as pl
#      
# img = io.imread("/dls/science/groups/das/ExampleData/SphereTestData/38644/recon_01830.tif")
#    
# watershed_segmentation(img, 3, 1, 1)
