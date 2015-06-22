def detect_circles_old(np_image, nb_circles):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pylab as pl
    
    from skimage import data, filter, color
    from skimage.transform import hough_circle
    from skimage.feature import peak_local_max
    from skimage.draw import circle_perimeter
    from skimage.util import img_as_ubyte
    
    pl.close('all')
    
    fig, ax = plt.subplots(1,1)
    
    # Load picture and detect edges
    
    # If elements of the image are |floats| > 1, normalise the image to be able to use img_as_ubyte
    if np_image.dtype=='float_' and not(np.array_equal(np_image>1, np.zeros(np_image.shape, dtype=bool))):
        image_norm = np_image/np.max(np_image)
    else:
        image_norm = np_image
    
    image = img_as_ubyte(image_norm)
    edges = filter.canny(image, sigma=3, low_threshold=10, high_threshold=50)
    
    # Detect radii
    
    step = 1
    
    hough_radii = np.arange(40, 210, step, np.int64)
    hough_res = hough_circle(edges, hough_radii)
    
    centers = []
    accums = []
    radii = []
    
    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract one circle
        peaks = peak_local_max(h, num_peaks=1)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius])
        
    # Draw the most prominent N circles (depends on how many circles we want to detect)
    image = color.gray2rgb(image)
    for idx in np.argsort(accums)[::-1][:nb_circles]:
        center_x, center_y = centers[idx]
        radius = radii[idx]
        cx, cy = circle_perimeter(center_y, center_x, radius)
        image[cy, cx] = (220, 20, 20) #each pixel of the perimeter is a triple that corresponds to a color
        
    ax.imshow(image, cmap=plt.cm.gray)
    ax.set_title('Circle detection on test image')
    
    #pl.show()
    
    return