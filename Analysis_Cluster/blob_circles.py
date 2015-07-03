def detect_circles(np_image):
    """
    Uses Hough transform to detect the radii and the
    centres of the "blobs" indicating the point of
    contact between the spheres
    """
    import numpy as np
    import pylab as pl
    from skimage.transform import hough_circle
    from skimage.feature import peak_local_max
    from skimage.draw import circle_perimeter
    
    pl.close('all')
    
    min_rad = int(max(np_image.shape[0], np_image.shape[1]) / 4.0)
    max_rad = int(max(np_image.shape[0], np_image.shape[1]) / 2.0)
    step = 1
     
    hough_radii = np.arange(min_rad, max_rad, step, np.float64)
    hough_res = hough_circle(np_image, hough_radii)
    
    centers = []
    accums = []
    radii = []
    circles = []  # to get the outlines of the circles
    C = []  # to get the centres of the circles, in relation to the different areas
    
    # For each radius, extract one circle
    for radius, h in zip(hough_radii, hough_res):
        peaks = peak_local_max(h, num_peaks=1)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius])
    
    for idx in np.argsort(accums)[::-1][:1]:
        center_x, center_y = centers[idx]
        C.append((center_x, center_y))
        radius = radii[idx]
        cx, cy = circle_perimeter(center_y, center_x, np.int64(radius))
        circles.append((cy, cx))
    
    #np_image[circles[0][0], circles[0][1]] = 0
    
    pl.imshow(np_image)
    pl.title('Circle detection on real image using Hough transform\n- optimised with image labelling algorithm -', fontdict={'fontsize': 20,'verticalalignment': 'baseline','horizontalalignment': 'center'})
    pl.colorbar()
    #pl.show()
        
    C_cp = C
    C = []
  
    if radius % 2 != 0:
        C.append((C_cp[0][0] + 0.5, C_cp[0][1] + 0.5))
    elif radius % 2 != 0:
        C.append((C_cp[0][0] + 0.5, C_cp[0][1]))
    elif radius % 2 != 0:
        C.append((C_cp[0][0], C_cp[0][1] + 0.5))
    else:
        C.append((C_cp[0][0], C_cp[0][1]))
  	
    return C

"""def add_noise(np_image, amount):
    import numpy as np
    import pylab as pl
    from scipy import misc
    
    noise = np.random.randn(np_image.shape[0],np_image.shape[1])
    norm_noise = noise/np.max(noise)
    np_image = np_image + norm_noise*np.max(np_image)*amount
    pl.imshow(np_image, cmap=pl.cm.Greys)
    pl.show()
    misc.imsave("noisy_circles.png",np_image)
    return np_image"""

#noisy = add_noise(img, 0.9)


"""radii = []
img = np.array(Image.open("noisy_circles.png").convert('L'))
rad = gr.plot_radii(img, (64,65.5))
radii.append(rad)

print np.mean(radii)"""