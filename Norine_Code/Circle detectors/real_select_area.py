def select_area(np_image):
    
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
    bw = dilation(bw, selem=square(3))
    
    # Fill the circles (used if some circles are split by a defect - unnecessary otherwise, take longer)
    
    label_image = label(bw, neighbors=4)
    labelCount = np.bincount(label_image.ravel())
    background = np.argmax(labelCount)
    
    for region in regionprops(label_image, ['Area', 'Label']):
        if region['Area'] < 90000:
            label_image[np.where(label_image==region['Label'])] = background
    
    bw[label_image != background] = 1
    dnp.plot.image(label_image)
    bw = erosion(bw, selem=square(9))
    
    # Remove artifacts connected to image border
    cleared = bw.copy()
    clear_border(cleared)
    
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
    
    print len(areas)
    
    #for i in range(len(areas)):
     #   pl.subplot(1,len(areas),i+1)
      #  pl.imshow(areas[i])
    #pl.show()
    
    return