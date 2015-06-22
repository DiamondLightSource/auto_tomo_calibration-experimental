def draw_areas(np_image, bord):
    
    import numpy as np
    import pylab as pl
    from skimage import color
    from skimage.util import img_as_ubyte
    
    # Convert the image in RGB
    
    if (np_image.dtype=='float_' or np_image.dtype=='float16' or np_image.dtype=='float32') and not(np.array_equal(np.absolute(np_image)>1, np.zeros(np_image.shape, dtype=bool))):
        np_image_norm = np_image/np.linalg.norm(np_image)
    else:
        np_image_norm = np_image
    
    new_image = img_as_ubyte(np_image_norm)
    new_image = color.gray2rgb(new_image)
    
    # Draw the areas on the whole image
    
    for i in range(len(bord)):
        new_image[bord[i][0]-1:bord[i][0]+1, bord[i][1]-1:bord[i][3]+1] = (220, 20, 20)
        new_image[bord[i][2]-1:bord[i][2]+1, bord[i][1]-1:bord[i][3]+1] = (220, 20, 20)
        new_image[bord[i][0]+1:bord[i][2]-1, bord[i][1]-1:bord[i][1]+1] = (220, 20, 20)
        new_image[bord[i][0]+1:bord[i][2]-1, bord[i][3]-1:bord[i][3]+1] = (220, 20, 20)
        
        print 'Area ' + repr(i+1) + ' : ' + repr(bord[i][2]-bord[i][0]) + ' x ' + repr(bord[i][3]-bord[i][1])
    
    pl.subplot(1,2,1)
    pl.imshow(new_image, cmap=pl.cm.YlOrRd, aspect='auto')
    pl.tight_layout()
    pl.colorbar()
    pl.subplot(1,2,2)
    pl.imshow(np_image, cmap=pl.cm.YlOrRd, aspect='auto')
    pl.tight_layout()
    pl.colorbar()
    pl.show()
    
    return

def select_area_for_detector(np_image, centroids):
    
    import numpy as np
    import pylab as pl
    
    from skimage import data
    from skimage.filter import threshold_otsu, sobel
    from skimage.segmentation import clear_border
    from skimage.morphology import label, closing, square
    from skimage.measure import regionprops
    
    # Apply threshold
    
    if (np_image.dtype=='float_' or np_image.dtype=='float16' or np_image.dtype=='float32') and not(np.array_equal(np.absolute(np_image)>1, np.zeros(np_image.shape, dtype=bool))):
        image_norm = np_image/np.linalg.norm(np_image)
    else:
        image_norm = np_image
    
    edges = sobel(image_norm)
    edges_closed = closing(edges, square(2))
    
    nbins = 50
    threshold = threshold_otsu(edges, nbins)
    
    bw = edges_closed >= threshold
    
    # Remove artifacts connected to image border
    
    cleared = bw.copy()
    clear_border(cleared)
    
    # Label image regions
    
    label_image = label(bw)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    
    areas = []
    bord = []
    
    # Extract information from the regions
    
    for region in regionprops(label_image, ['Area', 'BoundingBox', 'Centroid', 'Label']):
        
        # Skip wrong regions
        index = np.where(label_image==region['Label'])
        if index[0].size==0 & index[1].size==0:
            continue
        
        # Skip small images
        if region['Area'] < 2500:
            continue
        
        # Extract the regions
        minr, minc, maxr, maxc = region['BoundingBox']
        margin = len(np_image)/100
        bord.append([minr-margin, minc-margin, maxr+margin, maxc+margin])
        areas.append(np_image[minr-margin:maxr+margin,minc-margin:maxc+margin])
        
        # Extract the centres of regions
        cx, cy = region['Centroid']
        centroids.append((int(cx), int(cy)))
    
    return areas, bord

def detect_circles(np_image):
    
    import numpy as np
    import pylab as pl
    
    from skimage import filter
    from skimage.transform import hough_circle
    from skimage.feature import peak_local_max
    from skimage.draw import circle_perimeter
    from skimage.util import img_as_ubyte
    from skimage.morphology import closing, square
    from skimage.filter import threshold_otsu
    
    pl.close('all')
    
    centroids = [] # to get the centres of the circles, in relation to the whole image
    
    # Resize the image
    
    areas, bord = select_area_for_detector(np_image, centroids)
    print 'Length of areas : ' + repr(len(areas))
    
    # Find the biggest area
    shape_x = []
    for i in range(0, len(areas)):
        shape_x.append(areas[i].shape[0])
    max_x = max(shape_x)
    i_max = shape_x.index(max_x)
    bord_max = bord[i_max]
    
    # Resize
    np_image = areas[i_max]
    dnp.plot.image(np_image)
    
    '''Part with 1 select_areas --------------------------------'''
    '''
    # Check the areas
    
    index = []
    for i in range(0, len(areas)):
        # Jump out of bound areas and almost same areas than earlier (= same circle)
        if areas[i].shape[0] >= np_image.shape[0] or areas[i].shape[1] >= np_image.shape[1]\
        or bord[i][0] <= bord[i_max][0] or bord[i][2] >= bord[i_max][2] or bord[i][1] <= bord[i_max][1] or bord[i][3] >= bord[i_max][3]\
        or abs(bord[i][0]-bord[i-1][0]) <= 50 and abs(bord[i][1]-bord[i-1][1]) <= 50:
        #or abs(areas[i].shape[0] - areas[i-1].shape[0]) <= 10:
            index.append(i)
            continue
    
    areas = np.delete(areas, index)
    centroids = np.asarray(centroids)
    centroids = np.delete(centroids, index, 0)
    bord = np.delete(bord, index, 0)
    
    for i in range(len(bord)):
        print 'Area ' + repr(i+1) + ' : ' + repr(bord[i][2]-bord[i][0]) + ' x ' + repr(bord[i][3]-bord[i][1])
    '''
    '''------------------------------------'''
    
    '''Part with 2 select_areas --------------------------------'''
    
    # Select areas on rescaled image
    
    centroids = []
    areas, bord = select_area_for_detector(np_image, centroids)
    print 'Length of areas : ' + repr(len(areas))
    
    # Check the areas
    index = []
    for i in range(0, len(areas)):
        # Jump out of bound areas and almost same areas than earlier (= same circle)
        if areas[i].shape[0] >= 1000 or areas[i].shape[1] >= 1000\
        or areas[i].shape[0] <= 100 or areas[i].shape[1] <= 100\
        or abs(areas[i].shape[0] - areas[i-1].shape[0]) <= 10:
            index.append(i)
            continue
    
    areas = np.delete(areas, index)
    centroids = np.asarray(centroids)
    centroids = np.delete(centroids, index, 0)
    #bord = np.delete(bord, index, 0)
    #
    #for i in range(len(bord)):
    #    print 'Area ' + repr(i+1) + ' : ' + repr(bord[i][2]-bord[i][0]) + ' x ' + repr(bord[i][3]-bord[i][1])
    
    '''------------------------------------'''
    
    # Detect circles into each area
    
    circles = [] # to get the outlines of the circles
    C = [] # to get the centres of the circles, in relation to the different areas
    
    for i in range(0, len(areas)):
        
        # Load picture and detect edges
        
        # If elements of the image are |floats| > 1, normalise the image to be able to use img_as_ubyte
        if (areas[i].dtype=='float_' or areas[i].dtype=='float16' or areas[i].dtype=='float32') and not(np.array_equal(np.absolute(areas[i])>1, np.zeros(areas[i].shape, dtype=bool))):
            image_norm = areas[i]/np.linalg.norm(areas[i])
        else:
            image_norm = areas[i]
        
        edges = filter.sobel(image_norm)
        edges_closed = closing(edges, square(2))
        
        nbins = 50
        threshold = threshold_otsu(edges, nbins)
        
        edges_bin = edges_closed>=threshold
        
        # Detect circles
        
        min_rad = int(len(areas[i])/4)
        max_rad = int(len(areas[i])/2)
        step = 1
        
        hough_radii = np.arange(min_rad, max_rad, step, np.int64)
        hough_res = hough_circle(edges_bin, hough_radii)
        
        centers = []
        accums = []
        radii = []
        
        # For each radius, extract one circle
        for radius, h in zip(hough_radii, hough_res):
            peaks = peak_local_max(h, num_peaks=1)
            centers.extend(peaks)
            accums.extend(h[peaks[:, 0], peaks[:, 1]])
            radii.extend([radius])
        
        # Draw the most prominent N circles (depends on how many circles we want to detect) => here only 1 thanks to select_area
        
        for idx in np.argsort(accums)[::-1][:1]:
            center_x, center_y = centers[idx]
            C.append((center_x, center_y))
            radius = radii[idx]
            cx, cy = circle_perimeter(center_y, center_x, radius)
            circles.append((cx, cy))
    
    # Convert the whole image to RGB
    
    if (np_image.dtype=='float_' or np_image.dtype=='float16' or np_image.dtype=='float32') and not(np.array_equal(np.absolute(np_image)>1, np.zeros(np_image.shape, dtype=bool))):
        np_image_norm = np_image/np.linalg.norm(np_image)
    else:
        np_image_norm = np_image
    
    new_image = img_as_ubyte(np_image_norm)
    
    '''Part with 2 select_areas --------------------------------'''
    
    # Draw the circles on the whole image
    
    gap = np.asarray(centroids) - np.asarray(C) # gap between area reference frame and whole image reference frame 
    
    for i in range(len(areas)):
        new_image[circles[i][0] + gap[i][0], circles[i][1] + gap[i][1]] = 3
        # To thicken the line
        new_image[circles[i][0]-1 + gap[i][0], circles[i][1]-1 + gap[i][1]] = 3
        new_image[circles[i][0]-1 + gap[i][0], circles[i][1] + gap[i][1]] = 3
        new_image[circles[i][0]-1 + gap[i][0], circles[i][1]+1 + gap[i][1]] = 3
        new_image[circles[i][0] + gap[i][0], circles[i][1]-1 + gap[i][1]] = 3
        new_image[circles[i][0] + gap[i][0], circles[i][1]+1 + gap[i][1]] = 3
        new_image[circles[i][0]+1 + gap[i][0], circles[i][1]-1 + gap[i][1]] = 3
        new_image[circles[i][0]+1 + gap[i][0], circles[i][1] + gap[i][1]] = 3
        new_image[circles[i][0]+1 + gap[i][0], circles[i][1]+1 + gap[i][1]] = 3
    
    '''------------------------------------'''
    
    '''Part with 1 select_areas --------------------------------'''
    '''
    gap = np.zeros((len(areas), 2))
    for i in range(len(areas)):
        gap[i][0] = bord[i][0] - bord_max[0]
        gap[i][1] = bord[i][1] - bord_max[1]
        
    gap = gap.astype(np.int64, copy=False)
    
    for i in range(len(areas)):
        new_image[circles[i][0] + gap[i][0], circles[i][1] + gap[i][1]] = 3
        # To thicken the line
        new_image[circles[i][0]-1 + gap[i][0], circles[i][1]-1 + gap[i][1]] = 3
        new_image[circles[i][0]-1 + gap[i][0], circles[i][1] + gap[i][1]] = 3
        new_image[circles[i][0]-1 + gap[i][0], circles[i][1]+1 + gap[i][1]] = 3
        new_image[circles[i][0] + gap[i][0], circles[i][1]-1 + gap[i][1]] = 3
        new_image[circles[i][0] + gap[i][0], circles[i][1]+1 + gap[i][1]] = 3
        new_image[circles[i][0]+1 + gap[i][0], circles[i][1]-1 + gap[i][1]] = 3
        new_image[circles[i][0]+1 + gap[i][0], circles[i][1] + gap[i][1]] = 3
        new_image[circles[i][0]+1 + gap[i][0], circles[i][1]+1 + gap[i][1]] = 3
    '''
    '''------------------------------------'''
    
    pl.imshow(new_image, cmap=pl.cm.YlOrRd)
    pl.title('Circle detection on real image using Hough transform\n- optimised with image labelling algorithm -', fontdict={'fontsize': 20,'verticalalignment': 'baseline','horizontalalignment': 'center'})
    #optimised with image labelling algorithm
    pl.colorbar()
    pl.show()
    
    return