import numpy as np
import pylab as pl
from scipy import ndimage, misc

from skimage import measure
from skimage import exposure
from skimage.morphology import watershed
from skimage.filter import threshold_otsu, sobel, denoise_tv_chambolle
from skimage.exposure import rescale_intensity
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter

from circle_fit import leastsq_circle


def select_area_for_detector(np_image):
    """
    Takes image as an input and processes it:
    1. TV DENOISING FILTER
    2. RESCALE THE INTENSITY TO FLOAT (SOME FN'S NEED IT)
    3. ENCHANCE CONTRAST USING PERCENTILES 2 TO 99
    4. OTSU THRESHOLD
    5. EUCLIDEAN DISTANCE MAP
    6. GET MAXIMA FROM THE EDM - FOR MARKERS
    7. APPLY WATERSHED ALGORITHM
    
    THEN EXTRACT INFORMATION FROM THE SEGMENTED OBJECTS.
    FOR EVERY CROPPED OBJECT USE LEAST SQUARES, TO
    FIND A RADIUS ESTIMATE FOR THE HOUGH (FASTER) AND USE
    THE APPROXIMATE AREA FOR OUTLIER ELIMINATION.
    
    
    """
    pl.close('all')
    
    image_filtered = denoise_tv_chambolle(np_image, weight=0.005)
    
    float_img = rescale_intensity(image_filtered,
                                  in_range=(image_filtered.min(),
                                            image_filtered.max()),
                                  out_range='float')
    
    p2, p98 = np.percentile(float_img, (2, 99))
    equalize = exposure.rescale_intensity(float_img,
                                          in_range=(p2, p98),
                                          out_range='float')
    
    binary = equalize > threshold_otsu(equalize)
    
    distance = ndimage.distance_transform_edt(binary)
    local_maxi = peak_local_max(distance,
                                indices=False, labels=binary)
     
    markers = ndimage.label(local_maxi)[0]
     
    labeled = watershed(-distance, markers, mask=binary)
    
    areas = []
    radius_fit = []
    bords = []
    
    # Extract information from the regions
    for region in measure.regionprops(labeled, ['Area', 'BoundingBox']):
        
        # Extract the coordinates of regions
        # Margin used to go beyond the region if it
        # might be too tight on the object
        minr, minc, maxr, maxc = region.bbox
        margin = 10
        
        # Crop out the Watershed segments and obtain circle edges
        crop = equalize[minr-margin:maxr+margin,minc-margin:maxc+margin].copy()
        binary = crop > threshold_otsu(crop)
        crop = sobel(binary)
        
        # Get the coordinates of the circle edges
        coords = np.column_stack(np.nonzero(crop))
        X = np.array(coords[:, 0]) + minr - margin 
        Y = np.array(coords[:, 1]) + minc - margin

        # Fit a circle and compare measured circle area with
        # area from the amount of pixels to remove trash
        try:
            XC, YC, RAD, RESID = leastsq_circle(X, Y)
            if region.area * 1.3 > np.pi * RAD**2:

                radius_fit.append(round(RAD, 2))
                bords.append((minr - margin, minc - margin,
                              maxr + margin, maxc + margin))
                areas.append(crop)
        except:
            continue
    
    return [radius_fit, bords, areas, equalize]


def detect_circles(np_image, folder, task_id):

    pl.close('all')

    radius_fit, bord, areas, equalize\
     = select_area_for_detector(np_image)
     
     
    # Check if the areas are too big
    index = []
    size = max(np_image.shape[0] / 2, np_image.shape[1] / 2)

    for i in range(0, len(areas)):
        # Jump too big
        if areas[i].shape[0] >= size * 1.5 or areas[i].shape[1] >= size * 1.5:
            index.append(i)
    
    if index != []:
        areas[:] = [item for i, item in enumerate(areas) if i not in index]
        bord[:] = [item for i, item in enumerate(bord) if i not in index]
        radius_fit[:] = [item for i, item in enumerate(radius_fit) if i not in index]

    print 'Number of areas:', len(areas)

    circles = []  # to get the outlines of the circles
    C = []  # to get the centres of the circles, in relation to the different areas
    R = []  # to get radii
     
    for i in range(0, len(areas)):
        
        # Hough radius estimate
        hough_radii = np.arange(radius_fit[i] - radius_fit[i]/2.,
                                radius_fit[i] + radius_fit[i]/2.)
        
        # Apply Hough transform
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
             
        for idx in np.argsort(accums)[::-1][:1]:
            center_x, center_y = centers[idx]
            C.append((center_x + minr, center_y + minc))
            radius = radii[idx]
            R.append(radius)
            cx, cy = circle_perimeter(int(round(center_x + minr, 0)),
                                      int(round(center_y + minc, 0)),
                                      int(round(radius, 0)))
            circles.append((cx, cy))
            
    
    # Draw a dot on the centre  
    if C:
        for cent in C:
            xc, yc = cent
            equalize[int(xc), int(yc)] = 0
            equalize[int(xc)-5:int(xc)+5, int(yc)-5:int(yc)+5] = 0
            
    # Draw circle outlines
    if circles:
        for per in circles:
            e1, e2 = per
            equalize[e1, e2] = 0
    
    # Save objects for checking the detection
    # Save only the tenth slice
    if task_id % 10 == 0:
        misc.imsave(folder + 'labels%05i.jpg' % task_id, equalize)

    return [C, R, circles, bord]


def watershed_segmentation(image, folder, task_id):

    centroids, radius, edges, bords = detect_circles(image, folder, task_id)
    
    return [centroids, radius, edges, bords]