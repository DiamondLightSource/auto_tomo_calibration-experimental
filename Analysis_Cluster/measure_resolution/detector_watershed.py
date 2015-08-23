from skimage import exposure, img_as_ubyte, dtype_limits
import numpy as np
import pylab as pl

from skimage.color import gray2rgb
from skimage import measure, io
from scipy import ndimage, misc, optimize
from skimage.morphology import watershed, label, reconstruction, binary_erosion, disk
from skimage.filter import threshold_otsu, sobel, threshold_adaptive, prewitt, denoise_tv_chambolle, scharr
from scipy.ndimage.morphology import binary_opening, binary_closing, binary_fill_holes, binary_dilation
from scipy.ndimage.filters import gaussian_filter, median_filter
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.filter import canny
from sklearn.cluster import spectral_clustering
from circle_fit import leastsq_circle
from skimage.draw import circle_perimeter, set_color
from skimage.exposure import rescale_intensity
from skimage.filter.rank import enhance_contrast_percentile, enhance_contrast, otsu


def select_area_for_detector(np_image):
    
   
    pl.close('all')
    
    image_filtered = denoise_tv_chambolle(np_image, weight=0.005)
    binary = image_filtered.copy()
    mask1 = 0.0019 < image_filtered 
    mask2 = 0.000475 > image_filtered
    
    binary[mask1] = 0
    
    binary[mask2] = 0
    
    binary[binary > 0] = 1

        
        
    
    distance = ndimage.distance_transform_edt(binary)
    local_maxi = peak_local_max(distance,
                                indices=False, labels=binary)
     
    markers = ndimage.label(local_maxi)[0]
     
    labeled = watershed(-distance, markers, mask=binary)
    
    pl.subplot(2, 3, 2)
    pl.title("image_filtered")
    pl.imshow(image_filtered)
    pl.gray()
#     pl.subplot(2, 3, 1)
#     pl.title("rescale")
#     pl.imshow(rescale)
#     pl.subplot(2, 3, 4)
#     pl.title("edges")
#     pl.imshow(edges)
    pl.subplot(2, 3, 5)
    pl.title("binary")
    pl.imshow(binary)
    pl.subplot(2, 3, 6)
    pl.title("label")
    pl.imshow(labeled)
    pl.show()
    pl.close('all')

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
        margin = 3
        
        crop = image_filtered[minr-margin:maxr+margin,minc-margin:maxc+margin].copy()

        thresh = threshold_otsu(crop)
        binary = crop >= thresh
#         pl.imshow(binary)
#         pl.show()
        crop = sobel(binary)
        

        
        coords = np.column_stack(np.nonzero(crop))
        X = np.array(coords[:,0]) + minr - margin 
        Y = np.array(coords[:,1]) + minc - margin

        try:
            XC, YC, RAD, RESID = leastsq_circle(X, Y)
            if region.area * 1.3 > np.pi*(RAD+margin)**2:
                
                centroids_fit.append((round(XC, 4), round(YC, 4)))
                radius_fit.append(round(RAD, 2))
#                 edge_coords.append((X, Y))
                bords.append((minr - margin, minc - margin, maxr + margin, maxc + margin))
                areas.append(crop)
        except:
            continue
    
    return [centroids_fit, radius_fit, bords, areas, image_filtered]


def detect_circles(np_image, folder, task_id):
    
    import numpy as np
    import pylab as pl
    
    from skimage.transform import hough_circle
    from skimage.feature import peak_local_max
    from skimage.draw import circle_perimeter
    from skimage.util import img_as_ubyte
    from scipy import ndimage, misc, optimize

    pl.close('all')

    centroids_fit, radius_fit, bord, areas, rescale\
     = select_area_for_detector(np_image)
        
    print 'Number of areas before selection:', len(areas)
    # Check the areas
    index = []
    size = max(np_image.shape[0] / 2, np_image.shape[1] / 2)

    for i in range(0, len(areas)):
        # Jump too big or too small areas
        if areas[i].shape[0] >= size*1.5 or areas[i].shape[1] >= size*1.5:
        #or areas[i].shape[0] <= size/10 or areas[i].shape[1] <= size/10:
            index.append(i)
            continue
    
    if index != []:
        areas[:] = [item for i,item in enumerate(areas) if i not in index]
        bord[:] = [item for i,item in enumerate(bord) if i not in index]
        centroids_fit[:] = [item for i,item in enumerate(centroids_fit) if i not in index]
        radius_fit[:] = [item for i,item in enumerate(radius_fit) if i not in index]
#         edge_coords[:] = [item for i,item in enumerate(edge_coords) if i not in index]
        
    print 'Borders after selection:', bord
    print 'Number of areas:', len(areas)
    

    # Detect circles into each area
     
    circles = [] # to get the outlines of the circles
    C = [] # to get the centres of the circles, in relation to the different areas
    R = [] # to get radii
     
    for i in range(0, len(areas)):
          
        hough_radii = np.arange(radius_fit[i] - radius_fit[i]/2., radius_fit[i] + radius_fit[i]/2.)
        hough_res = hough_circle(areas[i], hough_radii)
             
        centers = []
        accums = []
        radii = []
        minr, minc, maxr, maxc = bord[i]
        
        # For each radius, extract one circle
        for radius, h in zip(hough_radii, hough_res):
            peaks = peak_local_max(h, num_peaks=2)
            centers.extend(peaks)
            accums.extend(h[peaks[:, 0], peaks[:, 1]])
            radii.extend([radius, radius])
        try:     
            for idx in np.argsort(accums)[::-1][:1]:
                center_x, center_y = centers[idx]
                C.append((center_x + minr, center_y + minc))
                radius = radii[idx]
                R.append(radius)
                cx, cy = circle_perimeter(int(round(center_x + minr,0)), int(round(center_y + minc,0)), int(round(radius,0)))
                circles.append((cx, cy))
        except:
            # If watershed segmentation failed
            C.append([])
            circles.append([])
            R.append([])
            
    if circles:
        for cent in C:
            xc, yc = cent
            rescale[int(xc), int(yc)] = 0
            rescale[int(xc)-5:int(xc)+5, int(yc)-5:int(yc)+5] = 0
            
    # Hough circles
    if C:
        import matplotlib.pyplot as plt
        for per in circles:
            e1, e2= per
            rescale[e1, e2] = 0
    
    pl.imshow(rescale)
    pl.gray()
    pl.show()

    if task_id % 3 == 0:
        misc.imsave(folder + 'labels%05i.jpg' % task_id, rescale)

    return [C, R, circles, bord]


def watershed_segmentation(image, smooth_size, folder, task_id):

    centroids, radius, edges, bords = detect_circles(image, folder, task_id)
    
    print centroids
    print radius
    
    return [centroids, radius, edges, bords]

    
from skimage import io
# img = io.imread("/dls/science/groups/das/ExampleData/SphereTestData/38644/recon_01200.tif")
img = io.imread("/dls/tmp/tomas_aidukas/scans_july16/cropped/50873/image_00450.tif")

watershed_segmentation(img, 3, 1, 1)