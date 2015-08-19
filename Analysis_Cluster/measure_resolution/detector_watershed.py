import numpy as np
import pylab as pl

from skimage import measure, io
from scipy import ndimage, misc, optimize
from skimage.morphology import watershed, label
from skimage.filter import threshold_otsu, sobel, canny, prewitt, denoise_tv_chambolle
from scipy.ndimage.morphology import binary_opening, binary_closing, binary_fill_holes, binary_erosion, binary_dilation
from scipy.ndimage.filters import gaussian_filter, median_filter
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from sklearn.cluster import spectral_clustering
from skimage.draw import circle_perimeter


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
    smoothed = gaussian_filter(smoothed, 3)
#     smoothed = denoise_tv_chambolle(image, weight=0.002)
#     misc.imsave(folder + 'smooth%05i.jpg' % task_id, smoothed)

    # TODO: what do with thresh?
    thresh = threshold_otsu(smoothed)
     
    binary = smoothed > thresh
     
     #     # Close up any small cracks
    bin_close = binary_closing(binary, np.ones((5, 5)), iterations=10)
    # Open the image by connecting small cracks and remove salt
    bin_open = binary_opening(bin_close, np.ones((5, 5)), iterations=10)
#     filled = binary_fill_holes(bin_open)

 
    # Fill the holes inside the circles
#     bin_fill = binary_fill_holes(bin_close, np.ones((5, 5)))
     
    distance = ndimage.distance_transform_edt(bin_open)
    local_maxi = peak_local_max(distance,
                                indices=False, labels=bin_open)
     
    markers = ndimage.label(local_maxi)[0]
     
    labeled = watershed(-distance, markers, mask=bin_open)

#     pl.subplot(2, 3, 1)
#     pl.title("graph")
#     pl.imshow(binary)
#     pl.gray()
#     pl.subplot(2, 3, 2)
#     pl.title("label_im")
#     pl.imshow(smoothed)
#     pl.subplot(2, 3, 3)
#     pl.title("closed")
#     pl.imshow(bin_close)
#     pl.subplot(2, 3, 4)
# #         pl.title("filled")
# #         pl.imshow(bin_fill)
#     pl.subplot(2, 3, 5)
#     pl.title("local_maxi")
#     pl.imshow(distance)
#     pl.subplot(2, 3, 6)
#     pl.title("label")
#     pl.imshow(labeled)
#     pl.show()
#     pl.close('all')
    
#         labels_rw = random_walker(bin_close, markers)
#         pl.imshow(labels_rw, interpolation='nearest')
#         pl.show()

    return labeled, binary


def watershed_segmentation(image, smooth_size, folder, task_id):
    
#     if np.unique(image)[0] == 0.:
#         return [[], []]
#     
    labels, denoised = preprocessing(image, smooth_size, folder, task_id)
    centroids, radius, edges, bords = centres_of_mass_2D(labels, folder, task_id, denoised)
    
    print centroids
    print radius
    
    return [centroids, radius, edges, bords]


def centres_of_mass_2D(image, folder, task_id, original):
    """
    Calculates centres of mass
    for all the labels
    """
    from circle_fit import leastsq_circle
    centroids_fit = []
    radius_fit = []
    edge_coords = []
    bords = []
    circles = []
    
    for info in measure.regionprops(image, ['coords ', 'Label', 'Area', 'equivalent_diameter', 'Centroid']): 
        
        if info['Area'] > image.shape[0] / 4.:
            
            minr, minc, maxr, maxc = info.bbox
            margin = 30
            crop = original[minr-margin:maxr+margin,minc-margin:maxc+margin].copy()
#             edges = sobel(np.pad(info.image, 30, mode="constant"))
            edges = sobel(crop)
            print minr, minc
# #             pl.imshow(edges)
# #             pl.gray()
# #             pl.show()
#             
            coords = np.column_stack(np.nonzero(edges))
            X = np.array(coords[:,0]) + minr-margin 
            Y = np.array(coords[:,1]) + minc-margin

            XC, YC, RAD, RESID = leastsq_circle(X, Y)
            
            cx, cy = circle_perimeter(int(round(XC,0)) + minr-margin , int(round(YC,0)) + minc-margin, int(round(RAD,0)))

            
            if info.area * 1.5 > np.pi*RAD**2:
                centroids_fit.append((round(XC, 4), round(YC, 4)))
                radius_fit.append(round(RAD, 2))
                edge_coords.append((cx, cy))
                bords.append((minr - 30, minc - 30, maxr + 30, maxc + 30))
    # Check the areas
    index = []
    print "before clearing number of circles was", len(centroids_fit)

    size = max(image.shape[0] / 2, image.shape[1] / 2)

    for i in range(0, len(centroids_fit)):
        # Jump too big or too small areas
        if (maxr + 60 - minr) <= size/5 or (maxc + 60 - minc) <= size/5:
            index.append(i)

    if index != []:
        centroids_fit[:] = [item for i,item in enumerate(centroids_fit) if i not in index]
        radius_fit[:] = [item for i,item in enumerate(radius_fit) if i not in index]
        edge_coords[:] = [item for i,item in enumerate(edge_coords) if i not in index]
        bords[:] = [item for i,item in enumerate(bords) if i not in index]
        
    for i in range(1, len(centroids_fit)):
        # Jump almost same areas (= same circle)
        if (abs(bords[i][0] - bords[i-1][0]) <= 100 and abs(bords[i][2] - bords[i-1][2]) <= 100):
            index.append(i)
            continue
    
    if index != []:
        centroids_fit[:] = [item for i,item in enumerate(centroids_fit) if i not in index]
        radius_fit[:] = [item for i,item in enumerate(radius_fit) if i not in index]
        edge_coords[:] = [item for i,item in enumerate(edge_coords) if i not in index]
        bords[:] = [item for i,item in enumerate(bords) if i not in index]

    print "after removal, number of circles is", len(centroids_fit)
    
    # Plot stuff
    if centroids_fit:
        for cent in centroids_fit:
            xc, yc = cent
            image[int(xc), int(yc)] = 0
            image[int(xc)-5:int(xc)+5, int(yc)-5:int(yc)+5] = 0
            
    if task_id % 30 == 0:
        misc.imsave(folder + 'labels%05i.jpg' % task_id, image)
#     pl.imshow(image)
#     pl.show()
    return [centroids_fit, radius_fit, edge_coords, bords]

img = io.imread("/dls/science/groups/das/ExampleData/SphereTestData/38644/recon_01230.tif")
                    
watershed_segmentation(img, 3, 1, 1)
