def draw_sphere(np_image, centre, radius, value):
    
    import numpy as np
    
    '''DON'T FORGET TO CHANGE N'''
    
    N = 900
    
    Xc = centre[0]
    Yc = centre[1]
    Zc = centre[2]
    
    Y, X, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
    mask = (((X - Xc)**2 + (Y - Yc)**2 + (Z - Zc)**2) < radius**2)
    
    np_image[mask] = value
    
    #for i in range(N):
    #    dnp.plot.image(np_image[:,:,i])
    
    return

def add_noise(np_image, amount):
    
    import numpy as np
    
    N = np_image.shape[2]
    
    for i in range(N):
        noise = np.random.randn(np_image.shape[0],np_image.shape[1])
        norm_noise = noise/np.max(noise)
        np_image[:,:,i] = np_image[:,:,i] + norm_noise*np.max(np_image[:,:,i])*amount
    
    for i in range(N):
        dnp.plot.image(np_image[:,:,i])
    
    return

def display(centres, radii):
    
    import numpy as np
    import pylab as pl
    from mpl_toolkits.mplot3d import Axes3D
    
    pl.close('all')
    
    '''DON'T FORGET TO CHANGE N'''
    
    N = 200
    
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    for i in range(len(radii)):
        x = radii[i] * np.outer(np.cos(u), np.sin(v)) + centres[i][0]
        y = radii[i] * np.outer(np.sin(u), np.sin(v)) + centres[i][1]
        z = radii[i] * np.outer(np.ones(np.size(u)), np.cos(v)) + centres[i][2]
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color='r')
    
    ax.set_xlim(0,N)
    ax.set_ylim(0,N)
    ax.set_zlim(0,N)
    pl.title('Test image - 3D')
    
    pl.show()
    
    return

def select_area_circle(np_image): # TO SELECT AREAS WITH CIRCLES IN EACH SLICE
    
    import numpy as np
    import pylab as pl
    import matplotlib.patches as mpatches
    
    from skimage.filter import threshold_otsu
    from skimage.segmentation import clear_border
    from skimage.morphology import label, closing, square
    from skimage.measure import regionprops
    
    # Apply threshold
    
    nbins = 50
    threshold = threshold_otsu(np_image, nbins)
    
    bw = closing(np_image >= threshold, square(3))
    
    # Remove artifacts connected to image border
    
    cleared = bw.copy()
    clear_border(cleared)
    
    # Label image regions
    
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    
    areas = []
    bord = []
    centroids = []
    
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
        areas.append(np_image[minr-margin:maxr+margin,minc-margin:maxc+margin].copy())
        
        # Extract centroids
        cx = np.mean([maxr, minr])
        cy = np.mean([maxc, minc])
        centroids.append((int(round(cx)), int(round(cy))))
    
    return areas, bord, centroids

def select_area_sphere(np_image): # to select the cubes containing the spheres
    
    import numpy as np
    import pylab as pl
    
    pl.close('all')
    
    areas_circles = []
    bord_circles = []
    centroids_sphere = []
    
    N = np_image.shape[2]
    
    # Extract the areas
    
    for slice in range(N):
        
        areas_circle, bord_circle, centroids = select_area_circle(np_image[:,:,slice])
        
        areas_circles.append(areas_circle)
        bord_circles.append(bord_circle)
        centroids_sphere.append(centroids)
    
    # Get the centre of each sphere
    
    centroids_sep = []
    for slice in range(N):
        for i in range(len(centroids_sphere[slice])):
            centroids_sep.append(centroids_sphere[slice][i])
    
    centroids = list(set(centroids_sep))
    
    # Sort out the centres per spheres
    
    nb_spheres = len(centroids)
    areas_spheres = [None] * nb_spheres
    slices_spheres = [None] * nb_spheres
    bords_spheres = [None] * nb_spheres
    
    for n in range(nb_spheres):
        sphere = []
        slices = []
        bords = []
        for slice in range(N):
            for i in range(len(centroids_sphere[slice])):
                if centroids_sphere[slice][i] == centroids[n]:
                    sphere.append(areas_circles[slice][i])
                    slices.append(slice)
                    bords.append(bord_circles[slice][i])
        areas_spheres[n] = sphere
        slices_spheres[n] = slices
        bords_spheres[n] = bords
    
    return areas_spheres, slices_spheres, bords_spheres

def detect_spheres(image):
    
    import numpy as np
    import pylab as pl
    from mpl_toolkits.mplot3d import Axes3D
    
    from skimage import filter, color
    from skimage.transform import hough_circle
    from skimage.feature import peak_local_max
    from skimage.draw import circle_perimeter
    from skimage.util import img_as_ubyte
    
    # Select the areas
    
    areas, slices, bords = select_area_sphere(image)
    
    # Detect circles into each area
    
    fig = pl.figure()
    ax = fig.gca(projection='3d')
    
    nb_spheres = len(areas)
    
    C_spheres = [] # to get the centres of the circles in each sphere
    R_spheres = [] # to get radii of circles in each sphere
    
    for n in range(nb_spheres):
        
        C = [] # to get the centres of the circles, in relation to the different areas
        R = [] # to get radii
        
        for i in range(0,len(areas[n])):
            
            circle = 0 # to get the outlines of the circles
            
            # Load picture and detect edges
            
            # If elements of the image are |floats| > 1, normalise the image to be able to use img_as_ubyte
            if (areas[n][i].dtype=='float_' or areas[n][i].dtype=='float16' or areas[n][i].dtype=='float32') \
            and not(np.array_equal(np.absolute(areas[n][i])>1, np.zeros(areas[n][i].shape, dtype=bool))):
                image_norm = areas[n][i]/np.max(areas[n][i])
                #print 'in if'
            else:
                image_norm = areas[n][i]
                #print 'in else'
            
            image = img_as_ubyte(image_norm)
            edges = filter.canny(image, sigma=3, low_threshold=10, high_threshold=50)
            
            # Detect circles
            
            min_rad = int(len(areas[n][i])/4)
            max_rad = int(len(areas[n][i])/2)
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
                C.append((center_x + bords[n][i][0], center_y + bords[n][i][2])) # coordinates of centres in whole image
                radius = radii[idx]
                R.append(radius)
                cy, cx = circle_perimeter(center_y, center_x, radius)
                circle = (cx, cy)
            
            if circle == 0: # check if a circle has not been detected
                continue
            
            ax.plot(circle[0] + bords[n][i][0], circle[1] + bords[n][i][2], slices[n][i])
        
        C_spheres.append(C)
        R_spheres.append(R)
    
    # Calculate approximate centres and radii
    
    centres_spheres = []
    radii_spheres = []
    
    for n in range(nb_spheres):
        
        # Average centres
        CxCy = np.mean(C_spheres[n], 0)
        Cz = np.mean([slices[n][0], slices[n][len(slices[n])-1]])
        centres_spheres.append((int(round(CxCy[0])), int(round(CxCy[1])), int(round(Cz))))
        
        # Calculate horizontal and vertical radii
        radius_horizontal = np.max(R_spheres[n])
        radius_vertical = (slices[n][len(slices[n])-1] - slices[n][0]) / 2
        radii_spheres.append([int(round(radius_horizontal)), int(round(radius_vertical))])
    
    print 'Centres:', centres_spheres
    print 'Radii:', radii_spheres
    
    ax.set_xlim(0,200)
    ax.set_ylim(0,200)
    ax.set_zlim(0,200)
    pl.title('Sphere detection on test image')
    
    pl.show()
    
    return centres_spheres, radii_spheres