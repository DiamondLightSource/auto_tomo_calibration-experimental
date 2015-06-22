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

def select_area_for_detector(np_image):
    
    import numpy as np
    import pylab as pl
    
    from skimage import data
    from skimage.filter import threshold_otsu, sobel
    from skimage.segmentation import clear_border
    from skimage.morphology import label, closing, square, dilation, erosion
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
    bw = dilation(bw, selem=square(3)) # to thicken the outlines for label
    
    # Fill the circles (used if some circles are split by a defect - unnecessary otherwise, take longer)
    
    label_image = label(bw, neighbors=4)
    labelCount = np.bincount(label_image.ravel())
    background = np.argmax(labelCount)
    
    # Clear the small areas by putting them in background
    for region in regionprops(label_image, ['Area', 'Label']):
        if region['Area'] < 20000:
            label_image[np.where(label_image==region['Label'])] = background
    
    bw[label_image != background] = 255
    bw = erosion(bw, selem=square(9))
    
    # Remove artifacts connected to image border
    
    cleared = bw.copy()
    clear_border(cleared)
    pl.imshow(cleared, cmap=pl.cm.gray)
    pl.show()
    # Label image regions
    
    label_image = label(bw, neighbors=4)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    #dnp.plot.image(label_image)
    areas = []
    bord_min = []
    
    # Extract information from the regions
    
    for region in regionprops(label_image, ['Area', 'BoundingBox', 'Label']):
        
        # Skip wrong regions
        index = np.where(label_image==region['Label'])
        if index[0].size==0 & index[1].size==0:
            continue
        
        # Skip small images
        if region['Area'] < 2500:
            continue
        
        # Extract the coordinates of regions
        minr, minc, maxr, maxc = region['BoundingBox']
        margin = len(np_image)/100
        bord_min.append((minr-margin, minc-margin))
        areas.append(np_image[minr-margin:maxr+margin,minc-margin:maxc+margin].copy())
    
    return areas, bord_min

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
    
    # Resize the image
    
    areas, bord_min = select_area_for_detector(np_image)
    
    # Find the biggest area
    shape_x = []
    for i in range(0, len(areas)):
        shape_x.append(areas[i].shape[0])
    max_x = max(shape_x)
    i_max = shape_x.index(max_x)
    
    # Resize
    np_image = areas[i_max]
    dnp.plot.image(np_image)
    
    # Select areas on rescaled image
    
    areas, bord_min = select_area_for_detector(np_image)
    print 'Length of areas : ' + repr(len(areas))
    print bord_min
    
    # Check the areas
    index = []
    for i in range(0, len(areas)):
        # Jump too big or too small areas
        if areas[i].shape[0] >= 1000 or areas[i].shape[1] >= 1000\
        or areas[i].shape[0] <= 100 or areas[i].shape[1] <= 100:
            index.append(i)
            continue
    for i in range(1, len(areas)):
        # Jump almost same areas than earlier (= same circle)
        if (abs(bord_min[i][0] - bord_min[i-1][0]) <= 10 and abs(bord_min[i][1] - bord_min[i-1][1]) <= 10):
            index.append(i)
            continue
    
    areas = np.delete(areas, index)
    bord_min = np.asarray(bord_min)
    bord_min = np.delete(bord_min, index, 0)
    print 'Length of areas : ' + repr(len(areas))
    print bord_min
    
    # Detect circles into each area
    
    circles = [] # to get the outlines of the circles
    C = [] # to get the centres of the circles, in relation to the different areas
    
    for i in range(0, len(areas)):
        
        # Load picture and detect edges
        
        # If elements of the image are |floats| > 1, normalise the image to be able to use img_as_ubyte
        if (areas[i].dtype=='float_' or areas[i].dtype=='float16' or areas[i].dtype=='float32') and not(np.array_equal(np.absolute(areas[i])>1, np.zeros(areas[i].shape, dtype=bool))):
            image_norm = areas[i] / np.linalg.norm(areas[i])
        else:
            image_norm = areas[i]
        
        edges = filter.sobel(image_norm)
        edges_closed = closing(edges, square(2))
        
        nbins = 50
        threshold = threshold_otsu(edges, nbins)
        
        edges_bin = edges_closed >= threshold
        
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
        
        # Find the most prominent N circles (depends on how many circles we want to detect) => here only 1 thanks to select_area
        for idx in np.argsort(accums)[::-1][:1]:
            center_x, center_y = centers[idx]
            C.append((center_x, center_y))
            radius = radii[idx]
            cx, cy = circle_perimeter(center_y, center_x, radius)
            circles.append((cy, cx))
    
    # Convert the whole image to RGB
    
    if (np_image.dtype=='float_' or np_image.dtype=='float16' or np_image.dtype=='float32') and not(np.array_equal(np.absolute(np_image)>1, np.zeros(np_image.shape, dtype=bool))):
        np_image_norm = np_image/np.linalg.norm(np_image)
    else:
        np_image_norm = np_image
    
    new_image = img_as_ubyte(np_image_norm)
    
    # Draw the circles on the whole image
    
    for i in range(len(areas)):
        new_image[circles[i][0] + bord_min[i][0], circles[i][1] + bord_min[i][1]] = 3
        # To thicken the line
        new_image[circles[i][0]-1 + bord_min[i][0], circles[i][1]-1 + bord_min[i][1]] = 3
        new_image[circles[i][0]-1 + bord_min[i][0], circles[i][1] + bord_min[i][1]] = 3
        new_image[circles[i][0]-1 + bord_min[i][0], circles[i][1]+1 + bord_min[i][1]] = 3
        new_image[circles[i][0] + bord_min[i][0], circles[i][1]-1 + bord_min[i][1]] = 3
        new_image[circles[i][0] + bord_min[i][0], circles[i][1]+1 + bord_min[i][1]] = 3
        new_image[circles[i][0]+1 + bord_min[i][0], circles[i][1]-1 + bord_min[i][1]] = 3
        new_image[circles[i][0]+1 + bord_min[i][0], circles[i][1] + bord_min[i][1]] = 3
        new_image[circles[i][0]+1 + bord_min[i][0], circles[i][1]+1 + bord_min[i][1]] = 3
    
    dnp.plot.image(new_image)
    pl.imshow(new_image, cmap=pl.cm.YlOrRd)
    pl.title('Circle detection on real image using Hough transform\n- optimised with image labelling algorithm -', fontdict={'fontsize': 20,'verticalalignment': 'baseline','horizontalalignment': 'center'})
    pl.colorbar()
    pl.show()
    
    return