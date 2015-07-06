def draw_areas(np_image, areas, bord):
    
    import numpy as np
    import pylab as pl
    from skimage.morphology import rectangle
    
    # Convert the image in RGB
    
    if (np_image.dtype=='float_' or np_image.dtype=='float16' or np_image.dtype=='float32') and not(np.array_equal(np.absolute(np_image)>1, np.zeros(np_image.shape, dtype=bool))):
        np_image_norm = np_image/np.linalg.norm(np_image)
    else:
        np_image_norm = np_image.copy()
    
    # Draw the areas on the whole image
    
    for i in range(len(areas)):
        lineX = np.arange(areas[i].shape[0])
        lineY = np.arange(areas[i].shape[1])
        
        np_image_norm[lineX + bord[i][0], bord[i][2]] = np.max(np_image_norm)
        np_image_norm[bord[i][0], lineY + bord[i][2]] = np.max(np_image_norm)
        # To thicken the line
        np_image_norm[lineX + bord[i][0]+1, bord[i][2]] = np.max(np_image_norm)
        np_image_norm[bord[i][0], lineY + bord[i][2]+1] = np.max(np_image_norm)
        np_image_norm[lineX + bord[i][0]-1, bord[i][2]] = np.max(np_image_norm)
        np_image_norm[bord[i][0], lineY + bord[i][2]-1] = np.max(np_image_norm)
        
        np_image_norm[lineX + bord[i][0], bord[i][2] + areas[i].shape[1]] = np.max(np_image_norm)
        np_image_norm[bord[i][0] + areas[i].shape[0], lineY + bord[i][2]] = np.max(np_image_norm)
        # To thicken the line
        np_image_norm[lineX + bord[i][0]+1, bord[i][2] + areas[i].shape[1]] = np.max(np_image_norm)
        np_image_norm[bord[i][0] + areas[i].shape[0], lineY + bord[i][2]+1] = np.max(np_image_norm)
        np_image_norm[lineX + bord[i][0]-1, bord[i][2] + areas[i].shape[1]] = np.max(np_image_norm)
        np_image_norm[bord[i][0] + areas[i].shape[0], lineY + bord[i][2]-1] = np.max(np_image_norm)
        
        # Plot the centre
        np_image_norm[bord[i][0] + int(areas[i].shape[0] / 2), bord[i][2] + int(areas[i].shape[1] / 2)] = np.max(np_image_norm)
        np_image_norm[bord[i][0] + int(areas[i].shape[0] / 2) + 1, bord[i][2] + int(areas[i].shape[1] / 2)] = np.max(np_image_norm)
        np_image_norm[bord[i][0] + int(areas[i].shape[0] / 2) - 1, bord[i][2] + int(areas[i].shape[1] / 2)] = np.max(np_image_norm)
        np_image_norm[bord[i][0] + int(areas[i].shape[0] / 2), bord[i][2] + int(areas[i].shape[1] / 2) + 1] = np.max(np_image_norm)
        np_image_norm[bord[i][0] + int(areas[i].shape[0] / 2), bord[i][2] + int(areas[i].shape[1] / 2) - 1] = np.max(np_image_norm)
        
        #print 'Area ' + repr(i+1) + ' : ' + repr(bord[i][2]-bord[i][0]) + ' x ' + repr(bord[i][3]-bord[i][1])
    
    pl.imshow(np_image_norm, cmap=pl.cm.YlOrRd)
    pl.colorbar()
    pl.show()
    
    return

def select_area_for_detector(np_image):
    
    import numpy as np
    import pylab as pl
    
    from skimage.filter import threshold_otsu, sobel
    from skimage.morphology import label
    from skimage.measure import regionprops
    from skimage.filter import denoise_tv_chambolle
    from skimage.feature import canny
    pl.close('all')
    
    # Find regions
    
    image_filtered = denoise_tv_chambolle(np_image, weight=0.002)
    edges = roberts(image_filtered)
    
    nbins = 50
    threshold = threshold_otsu(edges, nbins)
    edges_bin = edges >= threshold
    
    label_image = label(edges_bin)
    
    pl.imshow(edges)
    pl.show()
    
    areas = []
    areas_full = []
    bord = []
    
    # Extract information from the regions
    
    for region in regionprops(label_image, ['Area', 'BoundingBox', 'Label']):
        
        # Skip wrong regions
        index = np.where(label_image==region['Label'])
        if index[0].size==0 & index[1].size==0:
            continue
        
        # Skip small regions
        if region['Area'] < 100:
            continue
        
        # Extract the coordinates of regions
        minr, minc, maxr, maxc = region['BoundingBox']
        margin = len(np_image)/100
        bord.append((minr-margin, maxr+margin, minc-margin, maxc+margin))
        areas.append(edges_bin[minr-margin:maxr+margin,minc-margin:maxc+margin].copy())
        areas_full.append(np_image[minr-margin:maxr+margin,minc-margin:maxc+margin].copy())
    
    return areas, areas_full, bord

def detect_circles(np_image):
    
    import numpy as np
    import pylab as pl
    
    from skimage.transform import hough_circle
    from skimage.feature import peak_local_max
    from skimage.draw import circle_perimeter
    from skimage.util import img_as_ubyte
    
    pl.close('all')
    
    # Select areas
    
    areas, areas_full, bord = select_area_for_detector(np_image)
    
    # Check the areas
    index = []
    size = max(np_image.shape[0] / 2, np_image.shape[1] / 2)

    for i in range(0, len(areas)):
        # Jump too big or too small areas
        if areas[i].shape[0] >= size or areas[i].shape[1] >= size:
        #or areas[i].shape[0] <= size/5 or areas[i].shape[1] <= size/5:
            index.append(i)
            continue
    
    if index != []:
        areas[:] = [item for i,item in enumerate(areas) if i not in index]
        bord[:] = [item for i,item in enumerate(bord) if i not in index]
    
    index = []
    for i in range(1, len(areas)):
        # Jump almost same areas (= same circle)
        if (abs(bord[i][0] - bord[i-1][0]) <= 100 and abs(bord[i][2] - bord[i-1][2]) <= 100):
            index.append(i)
            continue
    
    if index != []:
        areas[:] = [item for i,item in enumerate(areas) if i not in index]
        bord[:] = [item for i,item in enumerate(bord) if i not in index]
    
    print 'Borders after selection:', bord
    print 'Number of areas:', len(areas)
    
    # Detect circles into each area
    
    circles = [] # to get the outlines of the circles
    C = [] # to get the centres of the circles, in relation to the different areas
    R = [] # to get radii
    
    for i in range(0, len(areas)):
        
        # Detect circles
        
        min_rad = int(max(areas[i].shape[0], areas[i].shape[1])/4)
        max_rad = int(max(areas[i].shape[0], areas[i].shape[1])/2)
        step = 1
        
        hough_radii = np.arange(min_rad, max_rad, step, np.int64)
        hough_res = hough_circle(areas[i], hough_radii)
        
        centers = []
        accums = []
        radii = []
        
        # For each radius, extract one circle
        for radius, h in zip(hough_radii, hough_res):
            peaks = peak_local_max(h, num_peaks=2)
            centers.extend(peaks)
            accums.extend(h[peaks[:, 0], peaks[:, 1]])
            radii.extend([radius,radius])
        
        # Find the most prominent N circles (depends on how many circles we want to detect) => here only 1 thanks to select_area
        for idx in np.argsort(accums)[::-1][:1]:
            center_x, center_y = centers[idx]
            C.append((center_x, center_y))
            radius = radii[idx]
            R.append(radius)
            cx, cy = circle_perimeter(center_y, center_x, radius)
            circles.append((cy, cx))
        
    """
    If the circle is an odd number of pixels wide, then that will displace the center by one pixel and give
    an uncertainty in the radius, producing the sine wave shape.
    
    THERE IS NO CLEAR WAY WHETHER TO SUBTRACT OR TO ADD THE HALF RADIUS
    
    C_cp = C
    C = []
    
    for i in range(len(areas)):
        try:
            circle_widthY = bord[i][2] - bord[i][0]
            circle_widthX = bord[i][3] - bord[i][1]

        except IndexError:
            return 0
        
        if circle_widthX % 2 != 0 and circle_widthY % 2 != 0:
            C.append((C_cp[i][0] + 0.5, C_cp[i][1] + 0.5))
        elif circle_widthX % 2 != 0:
            C.append((C_cp[i][0] + 0.5, C_cp[i][1]))
        elif circle_widthY % 2 != 0:
            C.append((C_cp[i][0], C_cp[i][1] + 0.5))
        else:
            C.append((C_cp[i][0], C_cp[i][1]))
    """
    return [bord, C, R, circles]

import h5py
import pylab as pl
import numpy as np

f = h5py.File('/dls/science/groups/i12/for Tomas/pco1-45808.hdf', 'r')
PATH = '/entry/instrument/detector/data/'
s = f[PATH]

slice = s[1000]

pl.imshow(slice)
pl.gray()
pl.show()

data = detect_circles(slice)