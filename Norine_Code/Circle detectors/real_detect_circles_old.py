def detect_circles_old(np_image, nb_circles):
    
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
    
    # Load picture and detect edges
    
    # If elements of the image are |floats| > 1, normalise the image to be able to use img_as_ubyte
    if (np_image.dtype=='float_' or np_image.dtype=='float16' or np_image.dtype=='float32') and not(np.array_equal(np.absolute(np_image)>1, np.zeros(np_image.shape, dtype=bool))):
        image_norm = np_image/np.linalg.norm(np_image)
    else:
        image_norm = np_image
    
    edges = filter.sobel(image_norm)
    edges_closed = closing(edges, square(2))
    
    nbins = 50
    threshold = threshold_otsu(edges, nbins)
    
    edges_bin = edges_closed >= threshold
    
    # Detect radii
    
    step = 1
    
    hough_radii = np.arange(150, 350, step, np.int64)
    hough_res = hough_circle(edges_bin, hough_radii)
    
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
    
    image = img_as_ubyte(image_norm)
    i = 0
    r = []
    while i < nb_circles:
        idx = np.argsort(accums)[::-1][:nb_circles][i]
        center_x, center_y = centers[idx]
        radius = radii[idx]
        r0 = np.asarray(r)
        if (np.absolute(r0-radius)<=1).any(): # jump if it is the radius of a circle already detected
            nb_circles += 1
            continue
        r.append(radius)
        cx, cy = circle_perimeter(center_y, center_x, radius)
        image[cy, cx] = 3
        i += 1
    
    pl.imshow(image, cmap=pl.cm.YlOrRd)
    pl.title('Circle detection on real image')
    pl.colorbar()
    pl.show()
    
    return