import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage import measurements
from scipy import optimize
import EqnLine as line

from skimage import io
from skimage import measure, color
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu, sobel
from skimage.filter import denoise_tv_chambolle
from skimage.util import img_as_ubyte
from scipy.ndimage.filters import median_filter

import pickle


def save_data(filename, data):
    import pickle
    print("Saving data")
    f = open(filename, 'w')
    pickle.dump(data, f)
    f.close()
    
    
def watershed_segmentation(image):

    
    #threshold
    image = median_filter(image, 5)

    filter = threshold_otsu(image)
    image = (image > filter) * 1
     
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image)
    
#     fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7))
#     ax0, ax1, ax2 = axes
#     
#     ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
#     ax0.set_title('Overlapping objects')
#     ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
#     ax1.set_title('Distances')
#     ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
#     ax2.set_title('Separated objects')
#     
#     for ax in axes:
#         ax.axis('off')
#     
#     fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
#                         right=1)
#     plt.show()
    
    

    
    return labels


def centres_of_mass_2D(image):
    """
    Calculates centres of mass
    for all the labels
    """
    centroids = []
    bords = []
    areas = []
    radius = []

    for info in measure.regionprops(image, ['Centroid', 'BoundingBox', 'equivalent_diameter']): 
        
        centre = info['Centroid']
        minr, minc, maxr, maxc = info['BoundingBox']
        D = info['equivalent_diameter']
    
        
        margin = 0
        
        radius.append((D / 2.0))
        bords.append((minr-margin, minc-margin, maxr+margin, maxc+margin))
        areas.append(image[minr-margin:maxr+margin,minc-margin:maxc+margin].copy())
        centroids.append(centre)
        
    return centroids, areas, bords, radius



def calc_R(x, y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()



def leastsq_circle_fit(image, centres, bords, radius):


    for i in range(len(centres)):
        
        # threshold
        area = image[i]
        filter = threshold_otsu(area)
        area = (area >= filter) * 1
        
        x = np.array([area[k][0] for k in range(len(area))])
        y = np.array([area[k][1] for k in range(len(area))])
        
        # get the centre estimate
        minr, minc, maxr, maxc = bords[i]
        centroid = (centres[i][0] - minr, centres[i][1] - minc)
        print centroid
        print radius[i]
        
        center, ier = optimize.leastsq(f, centroid, args=(x, y))
        
        xc, yc = center
        Ri = calc_R(x, y, *center)
        R = Ri.mean()
        
        print center
        print R
         
        a = plt.figure( facecolor='white')  #figsize=(7, 5.4), dpi=72,
        plt.axis('equal')
     
        theta_fit = np.linspace(-np.pi, np.pi, 180)
     
        x_fit = xc + R*np.cos(theta_fit)
        y_fit = yc + R*np.sin(theta_fit)
        plt.plot(x_fit, y_fit, 'b-' , label="fitted circle", lw=2)
        plt.plot([xc], [yc], 'bD', mec='y', mew=1)
        plt.xlabel('x')
        plt.ylabel('y')   
        # plot data
        plt.plot(x, y, 'r-.', label='data', mew=1)
      
        plt.legend(loc='best',labelspacing=0.1 )
        plt.grid()
        plt.title('Least Squares Circle')
        plt.show()


def leastsq_whole(image, centres):

    filter = threshold_otsu(image)
    image = (image >= filter) * 1
    
    x = np.array([image[k][0] for k in range(len(image))])
    y = np.array([image[k][1] for k in range(len(image))])
    
    for i in range(len(centres)):

        centroid = centres[i]
        print centroid
        
        center, ier = optimize.leastsq(f, np.asarray(centroid), args=(x, y))
        
        xc, yc = center
        Ri = calc_R(x, y, *center)
        R = Ri.mean()
        
        print center
        print R
         
        a = plt.figure( facecolor='white')  #figsize=(7, 5.4), dpi=72,
        plt.axis('equal')
     
        theta_fit = np.linspace(-np.pi, np.pi, 180)
     
        x_fit = xc + R*np.cos(theta_fit)
        y_fit = yc + R*np.sin(theta_fit)
        plt.plot(x_fit, y_fit, 'b-' , label="fitted circle", lw=2)
        plt.plot([xc], [yc], 'bD', mec='y', mew=1)
        plt.xlabel('x')
        plt.ylabel('y')   
        # plot data
        plt.plot(x, y, 'r-.', label='data', mew=1)
     
        plt.legend(loc='best',labelspacing=0.1 )
        plt.grid()
        plt.title('Least Squares Circle')
        plt.show()

def distance(c1, c2):
    return np.sqrt( (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def find_contact(centroids, radius, tol = 1):
    """
    Check all centre pairs and determine,
    based on their radii, if they are in contact
    or not
    """
    touch_pts = []
    centres = []
    N = len(centroids)
    for i in range(N - 1):
        for j in range(i + 1, N):
            
            c1 = centroids[i]
            c2 = centroids[j]
            r1 = radius[i]
            r2 = radius[j]
            
            D = r1 + r2
            L = distance(c1, c2)
            
#             if np.allclose(D, L, 0, 2):
            if abs(D - L) <= tol:
                touch_pt = ((c1[0] + c2[0]) / 2., (c1[1] + c2[1]) / 2.) 
                print c1, " ", c2, "are in contact"
                print touch_pt
                touch_pts.append(touch_pt)
                centres.append((c1, c2))
                
    return touch_pts, centres


def crop_box(image, touch_pt, centres, size = 30):
    """
    Crop a region around the touch point
    and perform Siemens star resolution
    analysis
    """
    crops = []
    slopes = []
    for i in range(len(touch_pt)):
        
        c1 = centres[i][0]
        c2 = centres[i][1]
        m = line.slope(c1, c2)
        crop = image[int(touch_pt[i][0]) - size:int(touch_pt[i][0]) + size, int(touch_pt[i][1]) - size:int(touch_pt[i][1]) + size]
        
        crops.append(crop)
        slopes.append(m)

    return crops, slopes


def watershed_3d(sphere):
    """
    Markers should be int8
    Image should be uint8
    """
   
    sphere = median_filter(sphere, 3)
    thresh = threshold_otsu(sphere)
    sphere = (sphere >= thresh) * 1
    sphere = sobel(sphere)
    
    size = (sphere.shape[0], sphere.shape[1], sphere.shape[2])
    
    marker = np.zeros(size, dtype=np.int16)
    pl.imshow(sphere[:,:,50])
    pl.show()
    # mark everything outside as background
    marker[5, :, :] = -1
    marker[size[0] - 5, :, :] = -1
    marker[:, :, 5] = -1
    marker[:, :, size[2] - 5] = -1
    marker[:, 5, :] = -1
    marker[:, size[1] - 5, :] = -1
    marker[:,0,0] = -1
    # mark everything inside as a sphere
    marker[size[0] / 2., size[1] / 2., size[2] / 2.] = 5

    result = measurements.watershed_ift(sphere.astype(dtype=np.uint16), marker)
    pl.imshow(result[:,:,50])
    pl.show()
    
    return result


def watershed_slicing(image):
    """
    Does the watershed algorithm slice by slice
    """
    
    
    N = len(image)
    slice_centroids = []
    slice_radius = []
    
    for i in range(N):
        
        slice = image[:, :, i]
        
        labels_slice = watershed_segmentation(slice)
        centroids, areas, bords, radius = centres_of_mass_2D(labels_slice)
        
        slice_centroids.append(centroids)
        slice_radius.append(radius)
#         if i > 49:
#             print centroids
#             pl.imshow(labels_slice)
#             pl.show()
        
    return slice_centroids, slice_radius


def draw_sphere():
    
    import numpy as np
    
    sphere = np.zeros((100, 100 ,100))
    N = 100
    radius1 = 20
    radius2 = 21
    centre1 = (30, 30, 50)
    centre2 = (30, 70, 50)

    Xc1 = centre1[0]
    Yc1 = centre1[1]
    Zc1 = centre1[2]
    
    Xc2 = centre2[0]
    Yc2 = centre2[1]
    Zc2 = centre2[2]
    
    Y, X, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
    mask1 = (((X - Xc1)**2 + (Y - Yc1)**2 + (Z - Zc1)**2) < radius1**2)
    mask2 = (((X - Xc2)**2 + (Y - Yc2)**2 + (Z - Zc2)**2) < radius2**2)
    sphere[mask1] = 1
    sphere[mask2] = 1
    
    return sphere


def fitfunc(p, x, y, z):
    x0, y0, z0, R = p
    return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)



def leastsq_sphere(sphere, centre, radius):
    """
    Fit a sphere
    """
    # Guess initial parameters
    coords = []
    for x in range(len(sphere)):
        for y in range(len(sphere)):
            for z in range(len(sphere)):
                coords.append([x, y, z])
    print x.T
    print y
    p0 = [centre[0], centre[1], centre[2], radius]
    
    errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[3]
    
    p1, flag = optimize.leastsq(errfunc, p0, args=(x, y, z))
    
    print p1
    
    return p1





sphere = draw_sphere()

#np.save("test_sphere.npy", sphere)

#image = io.imread("test_slice.tif")
sphere = np.load('sphere1.npy')
# centroids, radii = watershed_slicing(sphere)
# save_data("test_analysis/centroids.dat", centroids)
# save_data("test_analysis/radii.dat", radii)


labels = watershed_3d(sphere)
print measurements.center_of_mass(sphere, labels, 5)
# labels = watershed_segmentation(image)
#
# centroids, areas, bords, radius, radius2 = centres_of_mass_2D(labels)
# 
# # leastsq_circle_fit(areas, centroids, bords, radius)
# # leastsq_whole(image, centroids)
# touch, centres = find_contact(centroids, radius2)
# 
# crop_img, slopes = crop_box(image, touch, centres)
# 
# line.eqn_line(crop_img[0], slopes[0]) 