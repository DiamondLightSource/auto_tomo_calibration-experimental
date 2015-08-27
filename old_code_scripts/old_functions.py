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
from skimage.filter import threshold_otsu, sobel
from skimage.filter import denoise_tv_chambolle
from skimage.util import img_as_ubyte
from scipy.ndimage.filters import median_filter, gaussian_filter

import pickle


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



def crop_box_2D(image, touch_pt, centres, size = 30):
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
        m = slope(c1, c2)
        crop = image[int(touch_pt[i][0]) - size:int(touch_pt[i][0]) + size, int(touch_pt[i][1]) - size:int(touch_pt[i][1]) + size]
        
        crops.append(crop)
        slopes.append(m)

    return crops, slopes


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

########################################################################


def slope(pt1, pt2):
    """
    Calculate the slope
    """
    try:
        m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        return m
    except RuntimeWarning:
        # division by zero indicates no slope
        return 0


def interceipt(pt1, pt2):
    """
    Get interceipt
    y = mx + b
    """
    m = slope(pt1, pt2)
    b = pt1[1] - m * pt1[0]
    return b


def get_y(m, b, x):
    y = m * x + b
    return y


def distance_2D(c1, c2):
    return np.sqrt( (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def find_contact_2D(centroids, radius, tol = 1):
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
            L = distance_2D(c1, c2)
            
#             if np.allclose(D, L, 0, 2):
            if abs(D - L) <= tol:
                touch_pt = ((c1[0] + c2[0]) / 2., (c1[1] + c2[1]) / 2.) 
                print c1, " ", c2, "are in contact"
                print touch_pt
                touch_pts.append(touch_pt)
                centres.append((c1, c2))
                
    return touch_pts, centres

def touch_lines_2D(image, m):
    """
    goes along a line given a certain interceipt
    takes in another interceipt and goes along it again 
    registering all the pixel values
    """
    
#     pl.imshow(image)
#     pl.show()

    x = range(-image.shape[0], image.shape[0])
        
    pixels = []
    # for each interceipt
    for b in x:
        # draw lines parallel to
        # the line connecting centres
        cop = image.copy()
        line = []
        for i in range(image.shape[0]):
            y = get_y(m, b, i)
            try:
                line.append(cop[i, y])
#                 cop[i, y] = 0
#                 pl.imshow(cop, cmap = "Greys_r")
#                 pl.pause(0.001)except:
            except:
                continue
            
        if line:
            pixels.append(line)
        
    for i in range(len(pixels)):
        pl.close('all')
        pl.plot(pixels[i])
        pl.pause(0.5)
    
    return
