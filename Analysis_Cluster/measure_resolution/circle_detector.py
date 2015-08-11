def select_area_for_detector(image):
    
    import numpy as np
    import pylab as pl
    
    from skimage.filter import threshold_otsu, sobel
    from skimage.morphology import label
    from skimage.measure import regionprops
    from skimage.filter import denoise_tv_chambolle
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    from scipy import ndimage as ndi

    pl.close('all')
    
    # Find regions
    
    image_filtered = denoise_tv_chambolle(image, weight=0.002)
    edges = sobel(image_filtered.astype("int32"))
    
    nbins = 50
    threshold = threshold_otsu(edges, nbins)
    edges_bin = edges >= threshold
    
    label_image = label(edges_bin)
    
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
        margin = len(image) / 100
        bord.append((minr-margin, maxr+margin, minc-margin, maxc+margin))
        areas.append(image[minr-margin:maxr+margin,minc-margin:maxc+margin].copy())
        areas_full.append(image[minr-margin:maxr+margin,minc-margin:maxc+margin].copy())
    
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
        if areas[i].shape[0] >= size or areas[i].shape[1] >= size\
        or areas[i].shape[0] <= size/5 or areas[i].shape[1] <= size/5:
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
            radii.extend([radius, radius])
        
        # Find the most prominent N circles (depends on how many circles we want to detect) => here only 1 thanks to select_area
        for idx in np.argsort(accums)[::-1][:1]:
            center_x, center_y = centers[idx]
            C.append((center_x, center_y))
            radius = radii[idx]
            R.append(radius)
            cx, cy = circle_perimeter(center_y, center_x, radius)
            circles.append((cy, cx))

    return [bord, C, R, circles]
