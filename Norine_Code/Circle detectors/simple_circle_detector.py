def draw_circle(np_image, x, y, radius, value):
    
    import numpy as np
    
    mg = np.meshgrid(dnp.arange(np_image.shape[0]), dnp.arange(np_image.shape[1]))
    mask = (((mg[0] - y)**2 + (mg[1] - x)**2) < radius**2)
    np_image[mask] = value
    
    dnp.plot.image(np_image)
    
    return

#Careful: the initial image has to be saved before erasing circles
def erase_circle(np_init_image, np_image, x, y, radius):
    
    import numpy as np
    
    mg = np.meshgrid(dnp.arange(np_image.shape[0]), dnp.arange(np_image.shape[1]))
    mask = (((mg[0] - y)**2 + (mg[1] - x)**2) < radius**2)
    np_image[mask] = np_init_image[mask]
    
    dnp.plot.image(np_image)
    
    return

def add_noise(np_image, amount):
    
    import numpy as np
    
    noise = np.random.randn(np_image.shape[0],np_image.shape[1])
    norm_noise = noise/np.max(noise)
    np_image = np_image + norm_noise*np.max(np_image)*amount
    
    dnp.plot.image(np_image)
    
    return np_image

def select_area_for_detector(np_image, centroids, nbins):
    
    import numpy as np
    import matplotlib.patches as mpatches
    import math
    import pylab as pl
    
    from skimage import data
    from skimage.filter import threshold_otsu
    from skimage.segmentation import clear_border
    from skimage.morphology import label, closing, square
    from skimage.measure import regionprops
    
    # Apply threshold
    
    threshold = threshold_otsu(np_image, nbins)
    
    #print 'Threshold = ' + repr(threshold)
    #threshold = 1
    bw = closing(np_image >= threshold, square(3))
    #pl.subplot(1,3,1)
    #pl.imshow(bw)
    
    # Remove artifacts connected to image border
    
    cleared = bw.copy()
    clear_border(cleared)
    #pl.subplot(1,3,2)
    #pl.imshow(cleared)
    
    # Label image regions
    
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    
    #pl.subplot(1,3,3)
    #pl.imshow(label_image)
    
    #pl.show()
    
    areas = []
    
    # Extract information from the regions
    
    for region in regionprops(label_image, ['Area', 'BoundingBox', 'Centroid']):
        
        # Skip small images
        if region['Area'] < 100:
            continue
        
        # Extract the regions
        minr, minc, maxr, maxc = region['BoundingBox']
        margin = len(np_image)/100
        areas.append(np_image[minr-margin:maxr+margin,minc-margin:maxc+margin])
        
        # Extract the centres of regions
        cx, cy = region['Centroid']
        centroids.append((int(cx), int(cy)))
    
    return areas

def detect_circles(np_image):
    
    import numpy as np
    import pylab as pl
    
    from skimage import data, filter, color
    from skimage.transform import hough_circle
    from skimage.feature import peak_local_max
    from skimage.draw import circle_perimeter
    from skimage.util import img_as_ubyte
    
    pl.close('all')
    
    centroids = [] # to get the centres of the circles, in relation to the whole image
    circles = [] # to get the outlines of the circles
    C = [] # to get the centres of the circles, in relation to the different areas
    
    # Need to adapt nbins according to values of pixels
    nbins = 10
    areas = select_area_for_detector(np_image, centroids, nbins)
    
    # Detect circles into each area
    
    for i in range(0,len(areas)):
        
        # Load picture and detect edges
        
        # If elements of the image are |floats| > 1, normalise the image to be able to use img_as_ubyte
        if (areas[i].dtype=='float_' or areas[i].dtype=='float16' or areas[i].dtype=='float32') and not(np.array_equal(np.absolute(areas[i])>1, np.zeros(areas[i].shape, dtype=bool))):
            image_norm = areas[i]/np.max(areas[i])
            #print 'in if'
        else:
            image_norm = areas[i]
            #print 'in else'
        
        image = img_as_ubyte(image_norm)
        edges = filter.canny(image, sigma=3, low_threshold=10, high_threshold=50)
        
        # NOT EXACTLY THE SAME EDGES WITH NOISY AND NOT
        # Detect circles
        
        min_rad = int(len(areas[i])/4)
        max_rad = int(len(areas[i])/2)
        step = 1
        
        hough_radii = np.arange(min_rad, max_rad, step, np.int64)
        hough_res = hough_circle(edges, hough_radii)
        
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
        np_image_norm = np_image / np.max(np_image)
    else:
        np_image_norm = np_image
    
    new_image = img_as_ubyte(np_image_norm)
    new_image = color.gray2rgb(new_image)
    
    # Draw the circles on the whole image
    
    gap = np.array(len(areas)) # gap between area reference frame and whole image reference frame 
    gap = np.asarray(centroids) - np.asarray(C)
    
    for i in range(len(areas)):
        new_image[circles[i][0] + gap[i][0], circles[i][1] + gap[i][1]] = (220, 20, 20) # this tuple corresponds with red colour
        # To thicken the line
        new_image[circles[i][0]-1 + gap[i][0], circles[i][1]-1 + gap[i][1]] = (220, 20, 20)
        new_image[circles[i][0]-1 + gap[i][0], circles[i][1] + gap[i][1]] = (220, 20, 20)
        new_image[circles[i][0]-1 + gap[i][0], circles[i][1]+1 + gap[i][1]] = (220, 20, 20)
        new_image[circles[i][0] + gap[i][0], circles[i][1]-1 + gap[i][1]] = (220, 20, 20)
        new_image[circles[i][0] + gap[i][0], circles[i][1]+1 + gap[i][1]] = (220, 20, 20)
        new_image[circles[i][0]+1 + gap[i][0], circles[i][1]-1 + gap[i][1]] = (220, 20, 20)
        new_image[circles[i][0]+1 + gap[i][0], circles[i][1] + gap[i][1]] = (220, 20, 20)
        new_image[circles[i][0]+1 + gap[i][0], circles[i][1]+1 + gap[i][1]] = (220, 20, 20)
    
    pl.imshow(new_image, cmap=pl.cm.Greys)
    pl.title('Circle detection using Hough transform\n- optimised with image labelling algorithm -', fontdict={'fontsize': 20,'verticalalignment': 'baseline','horizontalalignment': 'center'})
    pl.colorbar(shrink=.92)
    pl.show()
    
    #return new_image