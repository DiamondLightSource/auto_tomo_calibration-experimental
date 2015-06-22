def select_area(np_image):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    from skimage import data
    from skimage.filter import threshold_otsu
    from skimage.segmentation import clear_border
    from skimage.morphology import label, closing, square
    from skimage.measure import regionprops
    from scipy import ndimage
    
    # Apply threshold
    
    nbins = 40
    threshold = threshold_otsu(np_image, nbins)
    print threshold
    bw = closing(np_image > threshold, square(3))
    bw = ndimage.median_filter(bw, 3)
    dnp.plot.image(bw)
    # Remove artifacts connected to image border
    
    cleared = bw.copy()
    clear_border(cleared)
    
    # Label image regions
    
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    dnp.plot.image(label_image)
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
    print len(areas)
    return