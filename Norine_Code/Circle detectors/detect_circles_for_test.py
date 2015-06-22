def detect_circles(np_image):
    
    runfile('C:\\Users\\eqp83935\\workspace\\data\\src\\select_area.py')
    
    import numpy as np
    import matplotlib.pyplot as plt
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
    
    areas = select_area(np_image, centroids)
    
    # Detect circles into each area
    
    for i in range(0,len(areas)):
        
        # Load picture and detect edges
        
        # If elements of the image are |floats| > 1, normalise the image to be able to use img_as_ubyte
        if areas[i].dtype=='float_' and not(np.array_equal(areas[i]>1, np.zeros(areas[i].shape, dtype=bool))):
            image_norm = areas[i]/np.max(areas[i])
        else:
            image_norm = areas[i]
        
        image = img_as_ubyte(image_norm)
        edges = filter.canny(image, sigma=3, low_threshold=10, high_threshold=50)
        
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
            circles.append((cx, cy))
        
    # Convert the whole image to RGB
    
    if np_image.dtype=='float_' and not(np.array_equal(np_image>1, np.zeros(np_image.shape, dtype=bool))):
        np_image_norm = np_image/np.max(np_image)
    else:
        np_image_norm = np_image
    
    new_image = img_as_ubyte(np_image_norm)
    new_image = color.gray2rgb(new_image)
    
    # Draw the circles on the whole image
    
    gap = np.array(len(areas)) # gap between area reference frame and whole image reference frame 
    gap = np.asarray(centroids) - np.asarray(C)
    
    for i in range(len(areas)):
        new_image[circles[i][0] + gap[i][0], circles[i][1] + gap[i][1]] = (220, 20, 20) # this tuple corresponds with red colour
    
    return new_image