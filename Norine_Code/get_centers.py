def get_centers(np_image, nb_circles):
    
    import numpy as np
    
    from skimage import data, filter
    from skimage.transform import hough_circle
    from skimage.feature import peak_local_max
    from skimage.util import img_as_ubyte
    
    # Load picture and detect edges
    
    # If elements of the image are |floats| > 1, normalise the image to be able to use img_as_ubyte
    if np_image.dtype=='float_' and not(np.array_equal(np_image>1, np.zeros(np_image.shape, dtype=bool))):
        image_norm = np_image/np.max(np_image)
    else:
        image_norm = np_image
    
    image = img_as_ubyte(image_norm)
    edges = filter.canny(image, sigma=3, low_threshold=10, high_threshold=50)
    
    # Detect two radii
    step = 1
    
    hough_radii = np.arange(40, 210, step, np.int64)
    hough_res = hough_circle(edges, hough_radii)
    
    centers = []
    accums = []
    C = []
    
    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract two circles
        peaks = peak_local_max(h, num_peaks=2)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        
    # Draw the most prominent N circles (depends on how many circles we want to detect)
    for idx in np.argsort(accums)[::-1][:nb_circles]:
        center_x, center_y = centers[idx]
        C.append((center_x, center_y))
    
    C.sort()
    
    return C