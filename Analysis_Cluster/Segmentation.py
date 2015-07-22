import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage import measurements
from scipy import optimize

from skimage import io
from skimage import measure, color
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu
from skimage.filter import denoise_tv_chambolle
from skimage.util import img_as_ubyte


def watershed_segmentation(image):

    
    #threshold
    image = denoise_tv_chambolle(image, weight=0.002)

    filter = threshold_otsu(image)
    image = (image >= filter) * 1
    
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
    diameter = []

    for info in measure.regionprops(image, ['Centroid', 'BoundingBox', 'equivalent_diameter']): 
        
        centre = info['Centroid']
        minr, minc, maxr, maxc = info['BoundingBox']
        D = info['equivalent_diameter']
        
        
        margin = 0
        
        diameter.append((D / 2.0))
        bords.append((minr-margin, minc-margin, maxr+margin, maxc+margin))
        areas.append(image[minr-margin:maxr+margin,minc-margin:maxc+margin].copy())
        centroids.append(centre)
        radius.append((maxr - minr) / 2.0)
        
    return centroids, areas, bords, radius, diameter



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
                
    return touch_pts

def crop_box(image, touch_pt):
    """
    Crop a region around the touch point
    and perform Siemens star resolution
    analysis
    """
    
    
    return
                
image = io.imread("test_slice.tif")

labels = watershed_segmentation(image)
centroids, areas, bords, radius, radius2 = centres_of_mass_2D(labels)

# leastsq_circle_fit(areas, centroids, bords, radius)
# leastsq_whole(image, centroids)
find_contact(centroids, radius2)
pl.imshow(image.T)
pl.gray()
pl.show()