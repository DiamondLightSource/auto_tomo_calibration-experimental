"""
Input:
    centres of the circles [array of arrays]
    radii of the circles [array of arrays]
Output:
    None
    
-> creates a 3D figure to store the spheres
-> generates arrays with angles 
-> generate polar coordinates of the sphere
-> plot the surfaces
"""
def display(centres, radii):
    
    import numpy as np
    import pylab as pl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    pl.close('all')
    
    '''DON'T FORGET TO CHANGE N'''
    
    N = 100
    
    fig = pl.figure()
    # chooses a 3D axes projection
    image_3d = fig.add_subplot(111, projection='3d')
    
    # linspace(start, stop, number of intervals)
    # arrays containing angles theta and phi
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    # x = r cosT sinP
    # y = r sinT sinP
    # z = r cosP
    # multiplies the outer product matrix by the radius
    # to define the sphere boundaries and shifts 
    for i in range(len(radii)):
        x = radii[i] * np.outer(np.cos(u), np.sin(v)) + centres[i][0]
        y = radii[i] * np.outer(np.sin(u), np.sin(v)) + centres[i][1]
        z = radii[i] * np.outer(np.ones(np.size(u)), np.cos(v)) + centres[i][2]
        image_3d.plot_surface(x, y, z, rstride=4, cstride=4, color='r')
        
    image_3d.set_xlim(0,N)
    image_3d.set_ylim(0,N)
    image_3d.set_zlim(0,N)
    pl.title('Test image - 3D')
    plt.savefig("./Test_Results/Test_image.png")
    #pl.show()
    
    return 


"""
Input:
    Image containing circles [np.array]
    
Outputs:
    areas containing the segmented circles (edges) [array of spheres;
    these arrays contain slices, and these slices are a 2D array]
    coordinates of the borders [array of tuples]
    centres of the segmented circles [array of tuples]


-> Denoise the image
-> obtain Sobel edges
-> threshold the image
-> label areas
-> obtain the regions with regionprops function

Problems:
    nbins depends on the image
    "small areas" are image dependent
    noise filter weight is image dependent
    centroids might not be precise
    neighbours in label() might need changing
    margin can "mess up" things
"""
def select_area_circle(np_image):
    
    import numpy as np
    import pylab as pl
    import matplotlib.patches as mpatches
    
    from skimage.filter import threshold_otsu, sobel
    from skimage.measure import regionprops, label
    from skimage.restoration import denoise_tv_chambolle
    
    pl.close('all')
    
    # Find regions
    
    image_filtered = denoise_tv_chambolle(np_image, weight=0.002)
    edges = sobel(image_filtered)
    
    nbins = 10
    threshold = threshold_otsu(edges, nbins)
    edges_bin = edges >= threshold
    
    label_image = label(edges_bin)
    
    areas = []
    bord = []
    centroids = []
    areas_full = []
    
    # Extract information from the regions
    
    for region in regionprops(label_image, ['Area', 'BoundingBox', 'Label']):
        
        # Skip wrong regions
        index = np.where(label_image==region['Label'])
        if index[0].size==0 & index[1].size==0:
            continue
        
        # Skip small regions
        """if region['Area'] < 100:
            continue"""
        
        # Extract the coordinates of regions
        """
        IF LIST INDICES MUST BE INTEGERS THEN MARGIN IS TOO BIG
        IF OUT OF BOUNDS THEN IT IS TOO SMALL
        """
        minr, minc, maxr, maxc = region['BoundingBox']
        margin = 5#np_image.shape[0]
        bord.append((minr-margin, maxr+margin, minc-margin, maxc+margin))
        areas.append(edges_bin[minr-margin:maxr+margin,minc-margin:maxc+margin].copy())
        areas_full.append(np_image[minr - margin:maxr + margin, minc - margin:maxc + margin].copy())

        # Extract centroids
        cx = np.mean([minr,maxr])
        cy = np.mean([minc,maxc])
        centroids.append((int(cx), int(cy)))
    
    return areas, bord, centroids, areas_full


"""
Input:
    takes in a 3D image [3D numpy array]
Output:
    an array containing circle data as arrays ie
    sphere information [array of a 2D arrays]
    
    number of slices for each sphere [array of 1D array]
    
    borders for circles inside each slice for each sphere [array of 2 arrays of tuples]
    
-> for each slice delete "wrong" circles
-> find which slices are not close together (belongs to separate sphere)
-> segment these spehres out
-> store the circular data of each slice of the sphere
    
"""
"""
def select_area_sphere(np_image): # to select the cubes containing the spheres
    
    import numpy as np
    import pylab as pl
    
    pl.close('all')
    
    areas_circles = []
    bord_circles = []
    centroids_sphere = []
    
    N = np_image.shape[2]
    
    for slice in range(N):
        areas_circle, bord_circle, centroids = select_area_circle(np_image[:,:,slice])

        size = max(np_image.shape[0] / 2, np_image.shape[1] / 2)
    
        index = []
        for i in range(0, len(areas_circle)):
            if areas_circle[i].shape[0] >= size or areas_circle[i].shape[1] >= size\
        or areas_circle[i].shape[0] <= size/3 or areas_circle[i].shape[1] <= size/3:
                index.append(i)
                continue
        
        if index != []:
            areas_circle[:] = [item for i,item in enumerate(areas_circle) if i not in index]
            bord_circle[:] = [item for i,item in enumerate(bord_circle) if i not in index]
            centroids[:] = [item for i,item in enumerate(centroids) if i not in index]
        
        index = []
        for i in range(1, len(areas_circle)):
            if (abs(bord_circle[i][0] - bord_circle[i-1][0]) <= 10 and abs(bord_circle[i][2] - bord_circle[i-1][2]) <= 10):
                index.append(i)
                continue
        
        if index != []:
            areas_circle[:] = [item for i,item in enumerate(areas_circle) if i not in index]
            bord_circle[:] = [item for i,item in enumerate(bord_circle) if i not in index]
            centroids[:] = [item for i,item in enumerate(centroids) if i not in index]
        
        if len(areas_circle) != 0:
            print 'Len(areas) after:', len(areas_circle)
            print 'Bords:', bord_circle
        
        areas_circles.append(areas_circle)
        bord_circles.append(bord_circle)
        centroids_sphere.append(centroids)
        
    centroids_sep = []
    for slice in range(N):
        for i in range(len(centroids_sphere[slice])):
            centroids_sep.append(centroids_sphere[slice][i])
            
    centroids_sep.sort()

    index = []
    for i in range(len(centroids_sep)):
        # CHANGE TO FIT DATA
        if abs(centroids_sep[i][0] - centroids_sep[i-1][0]) < 100 and abs(centroids_sep[i][1] - centroids_sep[i-1][1]) < 100:
            index.append(i)
            continue
    
    centroids = []
    centroids[:] = [ item for i,item in enumerate(centroids_sep) if i not in index ]
    
    print 'Nb of spheres:', len(centroids)
    print 'Centres of spheres:', centroids
    
    nb_spheres = len(centroids)
    areas_spheres = []
    slices_spheres = []
    bords_spheres = []
    
    for n in range(nb_spheres):
        
        sphere = []
        slices = []
        bords = []
        
        for slice in range(N):
            for i in range(len(centroids_sphere[slice])):
                '''TO BE CHANGED TO FIT DATA'''
                if (centroids[n][0] - 50 < centroids_sphere[slice][i][0] < centroids[n][0] + 50)\
                and (centroids[n][1] - 50 < centroids_sphere[slice][i][1] < centroids[n][1] + 50):
                    sphere.append(areas_circles[slice][i])
                    slices.append(slice)
                    bords.append(bord_circles[slice][i])
                    
        if len(sphere) > 0:
            areas_spheres.append(sphere)
            slices_spheres.append(slices)
            bords_spheres.append(bords)
    
    nb_spheres = len(areas_spheres)
    
    for n in range(nb_spheres):
        print 'Len(areas[', n, ']):', len(areas_spheres[n])
    
    return areas_spheres, slices_spheres, bords_spheres
"""
def select_area_sphere(np_image):
     
    import numpy as np
    import pylab as pl
     
    pl.close('all')
     
    areas_circles = []
    bord_circles = []
    centroids_sphere = []
    areas_full_circles = []
     
    N = np_image.shape[2]
     
    # Extract the areas
     
    for slice in range(N):
         
        areas_circle, bord_circle, centroids, areas_full = select_area_circle(np_image[:,:,slice])
         
        areas_circles.append(areas_circle)
        bord_circles.append(bord_circle)
        centroids_sphere.append(centroids)
        areas_full_circles.append(areas_full) 
    # Get the centre of each sphere
     
    centroids_sep = []
    for slice in range(N):
        for i in range(len(centroids_sphere[slice])):
            centroids_sep.append(centroids_sphere[slice][i])
     
    centroids = list(set(centroids_sep))
     
    # Sort out the 2D areas among spheres
     
    nb_spheres = len(centroids)
    areas_full_spheres = [None] * nb_spheres
    areas_spheres = [None] * nb_spheres # to stock the different areas with circles among slices for each sphere
    slices_spheres = [None] * nb_spheres # to stock the slices corresponding to the areas
    bords_spheres = [None] * nb_spheres # to stock the borders of the 2D areas
     
    for n in range(nb_spheres):
        sphere = []
        slices = []
        bords = []
        sphere_full = []
        for slice in range(N):
            for i in range(len(centroids_sphere[slice])):
                if centroids_sphere[slice][i] == centroids[n]:
                    sphere.append(areas_circles[slice][i])
                    sphere_full.append(areas_full_circles[slice][i])
                    slices.append(slice)
                    bords.append(bord_circles[slice][i])
        areas_spheres[n] = sphere
        slices_spheres[n] = slices
        bords_spheres[n] = bords
        areas_full_spheres[n] = sphere_full
     
    return areas_spheres, slices_spheres, bords_spheres, areas_full_spheres


"""
Input:
    image to be analysed [3D array]
Output:
    cubes containing the spheres
    centres_spheres has the centres of the spheres
    radii_spheres has the radii of the spheres
    
-> uses select_area_sphere to obtain segmented spheres
-> loop through the spheres and for each sphere loop
    through the slices
-> apply Hough transform on each of these slices
-> generate arrays to store the info of the slices
    inside the sphere from Hough

-> loop through each sphere
-> find max dimension to define the cube size
-> get the borders for the cube
-> create a 3D array for the cube to store the spheres
-> get the centers of the circles relative to the cubes
"""
def detect_spheres(image):
    
    import numpy as np
    import pylab as pl
    from mpl_toolkits.mplot3d import Axes3D
    
    from skimage import filter, color
    from skimage.transform import hough_circle
    from skimage.feature import peak_local_max
    from skimage.draw import circle_perimeter, circle
    from skimage.util import img_as_ubyte
    
    # Select the areas
    
    areas, slices, bords, areas_full = select_area_sphere(image)
    # Detect circles into each area
    
    fig = pl.figure()
    ax = fig.gca(projection='3d')
    
    nb_spheres = len(areas)
    
    """
    find the largest dimension of all the areas
    """
    largest_areas = []
    largest_slice = []
    # get max area size
    # for each sphere...
    for n in range(nb_spheres):
        
        areas_3d = areas[n]
        largest = 0
        # for each sliced area...
        for i in range(len(areas_3d)):
            # get the cropped sphere slice
            slice_area = max(areas_3d[i].shape)
            if slice_area > largest:
                largest = slice_area
        
        largest_areas.append(largest-1)
    
    C_spheres = [] # to get the centres of the circles in each sphere
    R_spheres = [] # to get radii of circles in each sphere
    reconstructed_spheres = [] # stores the slices of each reconstructed sphere
    absolute_centers = []
    
    print largest_areas
    
    for n in range(nb_spheres):
        
        circles = []
        C = [] # to get the centres of the circles, in relation to the different areas
        R = [] # to get radii
        
        # largest_area gives the correct radii plots
        cube_dim = largest_areas[n]
        circle_slice = np.zeros((cube_dim,cube_dim,cube_dim-10))
        
        
        for i in range(0,len(areas[n])):
            
            circle_per = 0 # to get the outlines of the circles
            
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
            edges = image
            'filter.canny(image, sigma=3, low_threshold=10, high_threshold=50)'
            
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
                circle_per = (cx, cy)
                # generate a full circle - not perimeter
                # YOU CAN HAVE AN ARRAY OF ONE WITH PERIMETER
                # OR YOU CAN HAVE AN ARRAY OF ZEROS WITH CIRCLE
                ccy, ccx = circle(center_y, center_x, radius)
                circle_full = (ccx, ccy)

            if circle_per == 0: # check if a circle has not been detected
                continue
            
            ax.plot(circle_per[0] + bords[n][i][0], circle_per[1] + bords[n][i][2], slices[n][i])
            print "index i:", i, "border X:", bords[n][i][0], "center X", center_x, "center shift", largest_areas[n]/2 - center_x
            print "index i:", i, "border Y:", bords[n][i][2], "center Y", center_y, "center shift", largest_areas[n]/2 - center_y
            # largest_areas[n]/2.0 - center_y gets the center position for all of the segmented areas
            # center+border gives the absolute center in the image
            circle_slice[circle_full[0] + largest_areas[n]/2 - center_x -1, circle_full[1] + largest_areas[n]/2 - center_y -1 , slices[n][i] -min(slices[n])] = 255
            #circle_slice[circle_full[0] + bords[n][i][0], circle_full[1] + bords[n][i][2], slices[n][i]-min(slices[n])] = 255
            
            
        C_spheres.append(C)
        R_spheres.append(R)
        reconstructed_spheres.append(circle_slice)
        # Minus the margin
        absolute_centers.append((largest_areas[n]/2,largest_areas[n]/2,largest_areas[n]/2 - 5))
        
    # Calculate approximate centres and radii
    
    centres_spheres = []
    radii_spheres = []
    
    for n in range(nb_spheres):
        
        # Average centres
        CxCy = np.mean(C_spheres[n], 0)
        Cz = np.mean([slices[n][0], slices[n][len(slices[n]) - 1]])
        centres_spheres.append((int(round(CxCy[0])), int(round(CxCy[1])), int(round(Cz))))
        
        # Calculate horizontal and vertical radii
        radius_horizontal = np.max(R_spheres[n])
        radius_vertical = (slices[n][len(slices[n]) - 1] - slices[n][0]) / 2
        radii_spheres.append([int(round(radius_horizontal)), int(round(radius_vertical))])
        
    print 'Centres:', centres_spheres
    print 'Radii:', radii_spheres

        
    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.set_zlim(0,100)
    pl.title('Sphere detection on test image')
    plt.savefig("./Test_Results/Reconstructed_3D.png")
    #pl.show()
    
    return centres_spheres, radii_spheres, reconstructed_spheres, absolute_centers

"""
Input:
    Center coordinates [tuple]
    Radius coordinates [number]
    Value of the sphere's pixels [number]
Output:
    3D numpy image [numpy array]
"""
def draw_sphere(centre, radius, value):
    
    import numpy as np
    
    N = 100
    
    np_image = np.zeros((N, N, N))
    
    for i in xrange(len(radius)):
        
        
        Xc = centre[i][0]
        Yc = centre[i][1]
        Zc = centre[i][2]
        
        X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
        mask = (((X - Xc)**2 + (Y - Yc)**2 + (Z - Zc)**2) < radius[i]**2)
        
        np_image[mask] = value
    
    return np_image
    
"""
generate test spheres
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image

centre = [(30,30,30), (71,71,71)]#, (151,151,151)]
radius = [10,15]#,30,40]
value = 20

display(centre,radius)
img = draw_sphere(centre,radius,value)

img = np.asarray(img)

# sphere detection fn
cent_3d, rad_3d, recon, abs_cent = detect_spheres(img)

np.save("./Numpy_Files/reconstructed_spheres", recon)
np.save("./Numpy_Files/cent_3d", cent_3d)
np.save("./Numpy_Files/rad_3d", rad_3d)
np.save("./Numpy_Files/original_3d", img)
np.save("./Numpy_Files/abs_centers", abs_cent)

