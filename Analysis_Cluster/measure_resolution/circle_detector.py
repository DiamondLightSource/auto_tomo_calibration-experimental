import numpy as np
import pylab as pl

from skimage import measure, io
from scipy import ndimage, misc, optimize
from skimage.morphology import watershed, label, reconstruction, binary_erosion
from skimage.filter import threshold_otsu, sobel, canny, prewitt, denoise_tv_chambolle
from scipy.ndimage.morphology import binary_opening, binary_closing, binary_fill_holes, binary_dilation
from scipy.ndimage.filters import gaussian_filter, median_filter
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from sklearn.cluster import spectral_clustering
from circle_fit import leastsq_circle
from skimage.draw import circle_perimeter
from skimage.exposure import rescale_intensity


def circle_coordinates(cx, cy, r):
    """
    Get the circle perimeter coordinates
    to a floating point precision
    """
    X = []
    Y = []
    
    for theta in np.arange(0, 360, 0.5):
        x = np.cos(np.radians(theta)) * r + cx
        y = np.sin(np.radians(theta)) * r + cy
        
        X.append(x)
        Y.append(y)
        
    return X, Y



def select_area_for_detector(np_image):
    
   
    pl.close('all')
    
    # Find regions
    
    image_filtered = denoise_tv_chambolle(np_image, weight=0.002)
#     thresh = threshold_otsu(image_filtered)     
#     binary = image_filtered > thresh
     
    seed = np.copy(image_filtered)
    seed[1:-1, 1:-1] = image_filtered.max()
    mask = image_filtered
    eroded = reconstruction(seed, mask, method='erosion')
    
    thresh = threshold_otsu(eroded)
    binary = eroded > thresh
    # Open the image by connecting small cracks and remove salt
#     bin_open = binary_opening(binary, np.ones((3, 3)), iterations=5)

    distance = ndimage.distance_transform_edt(binary)
    local_maxi = peak_local_max(distance,
                                indices=False, labels=binary)
     
    markers = ndimage.label(local_maxi)[0]
     
    labeled = watershed(-distance, markers, mask=binary)
    
    
#     pl.subplot(2, 3, 1)
#     pl.title("filt")
#     pl.imshow(image_filtered)
#     pl.gray()
#     pl.subplot(2, 3, 2)
#     pl.title("binary")
#     pl.imshow(binary)
#     pl.subplot(2, 3, 3)
#     pl.title("eroded")
#     pl.imshow(eroded)
# #     pl.subplot(2, 3, 4)
# #     pl.title("dilated")
# #     pl.imshow(dilated)
# #     pl.subplot(2, 3, 5)
# #     pl.title("hdome")
# #     pl.imshow(hdome)
#     pl.subplot(2, 3, 6)
#     pl.title("label")
#     pl.imshow(labeled)
#     pl.show()
#     pl.close('all')

    areas = []
    centroids_fit = []
    radius_fit = []
    edge_coords = []
    bords = []
    
    
    # Extract information from the regions
    
    for region in measure.regionprops(labeled, ['Area', 'BoundingBox', 'Label']):
        
        # Skip wrong regions
        index = np.where(labeled==region['Label'])
        if index[0].size==0 & index[1].size==0:
            continue
        
        # Skip small regions
        if region['Area'] < 100:
            continue
        
        # Extract the coordinates of regions
        minr, minc, maxr, maxc = region.bbox
        margin = 30
        
#         crop = eroded[minr-margin:maxr+margin,minc-margin:maxc+margin].copy()
#         crop = sobel(crop)
        crop = sobel(np.pad(region.image, margin, mode="constant"))
        
        coords = np.column_stack(np.nonzero(crop))
        X = np.array(coords[:,0]) + minr - margin 
        Y = np.array(coords[:,1]) + minc - margin

        
        try:
            XC, YC, RAD, RESID = leastsq_circle(X, Y)
#             cx, cy = circle_perimeter(int(round(XC,0)), int(round(YC,0)), int(round(RAD,0)))
            cx, cy = circle_coordinates(XC, YC, RAD)
            if region.area * 1.5 > np.pi*RAD**2:
                centroids_fit.append((round(XC, 4), round(YC, 4)))
                radius_fit.append(round(RAD, 2))
                edge_coords.append((X, Y, cx, cy))
                bords.append((minr - margin, minc - margin, maxr + margin, maxc + margin))
                
                areas.append(crop)
        except:
            continue
    
    return [centroids_fit, radius_fit, edge_coords, bords, areas]


def detect_circles(np_image, folder, task_id):
    
    import numpy as np
    import pylab as pl
    
    from skimage.transform import hough_circle
    from skimage.feature import peak_local_max
    from skimage.draw import circle_perimeter
    from skimage.util import img_as_ubyte
    from scipy import ndimage, misc, optimize

    pl.close('all')
    
    # Select areas
    
    centroids_fit, radius_fit, edge_coords, bord, areas = select_area_for_detector(np_image)
    
    # Check the areas
    index = []
    size = max(np_image.shape[0] / 2, np_image.shape[1] / 2)

    for i in range(0, len(areas)):
        # Jump too big or too small areas
        if areas[i].shape[0] >= size*1.5 or areas[i].shape[1] >= size*1.5\
        or areas[i].shape[0] <= size/5 or areas[i].shape[1] <= size/5:
            index.append(i)
            continue
    
    if index != []:
        areas[:] = [item for i,item in enumerate(areas) if i not in index]
        bord[:] = [item for i,item in enumerate(bord) if i not in index]
        centroids_fit[:] = [item for i,item in enumerate(centroids_fit) if i not in index]
        radius_fit[:] = [item for i,item in enumerate(radius_fit) if i not in index]
        edge_coords[:] = [item for i,item in enumerate(edge_coords) if i not in index]
        
    index = []
    for i in range(1, len(areas)):
        # Jump almost same areas (= same circle)
        if (abs(bord[i][0] - bord[i-1][0]) <= 100 and abs(bord[i][2] - bord[i-1][2]) <= 100):
            index.append(i)
            continue
    
    if index != []:
        areas[:] = [item for i,item in enumerate(areas) if i not in index]
        bord[:] = [item for i,item in enumerate(bord) if i not in index]
        centroids_fit[:] = [item for i,item in enumerate(centroids_fit) if i not in index]
        radius_fit[:] = [item for i,item in enumerate(radius_fit) if i not in index]
        edge_coords[:] = [item for i,item in enumerate(edge_coords) if i not in index]
        
    print 'Borders after selection:', bord
    print 'Number of areas:', len(areas)
    

    # Detect circles into each area
     
    circles = [] # to get the outlines of the circles
    C = [] # to get the centres of the circles, in relation to the different areas
    R = [] # to get radii
     
#     for i in range(0, len(areas)):
#            
#         rad_fit = radius_fit[i]
#         print rad_fit
#         hough_radii = np.arange(rad_fit-20, rad_fit+ 20)
#         hough_res = hough_circle(areas[i], hough_radii)
#             
#         centers = []
#         accums = []
#         radii = []
#         minr, minc, maxr, maxc = bord[i]
#         # For each radius, extract one circle
#         for radius, h in zip(hough_radii, hough_res):
#             peaks = peak_local_max(h, num_peaks=2)
#             centers.extend(peaks)
#             accums.extend(h[peaks[:, 0], peaks[:, 1]])
#             radii.extend([radius, radius])
#             
#         # Find the most prominent N circles (depends on how many circles we want to detect) => here only 1 thanks to select_area
#         for idx in np.argsort(accums)[::-1][:1]:
#             center_x, center_y = centers[idx]
#             C.append((center_x + minr, center_y + minc))
#             radius = radii[idx]
#             R.append(radius)
#             cx, cy = circle_perimeter(int(round(center_x,0)) + minr , int(round(center_y,0)) + minc, int(round(radius,0)))
#             circles.append((cx, cy))

    if centroids_fit:
        for cent in centroids_fit:
            xc, yc = cent
            np_image[int(xc), int(yc)] = 0
            np_image[int(xc)-5:int(xc)+5, int(yc)-5:int(yc)+5] = 0
            
    if task_id % 30 == 0:
        misc.imsave(folder + 'labels%05i.jpg' % task_id, np_image)

    return [centroids_fit, radius_fit, edge_coords, bord]


def watershed_segmentation(image, smooth_size, folder, task_id):

    centroids, radius, edges, bords = detect_circles(image, folder, task_id)
    
    print centroids
    print radius
    
    return [centroids, radius, edges, bords]

# from skimage import io
# img = io.imread("/dls/science/groups/das/ExampleData/SphereTestData/38644/recon_01230.tif")
#                        
# watershed_segmentation(img, 3, 1, 1)