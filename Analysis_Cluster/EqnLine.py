import numpy as np
import pylab as pl
from scipy import optimize

from peak_detect import *


################### FITTING FN ################################

def gaussian(x, height, center, width):
    try:
        return height*np.exp(-(x - center)**2/(2*width**2))
    except:
        return 0


def fit_gaussian_to_signal(points, sigma = 3, rad_pos = 0., height = 0., width_guess = 10.):
    """
    Detects peaks
    First guess for the Gaussian is the firs maxima in the signa;
    """
    rms_noise5 = rms_noise_signal(points) * 3
    peaks = peakdetect(points,
                       lookahead = 15)#, delta = rms_noise5)
    try:
        centre_guess1 = peaks[1][0][0]
        print peaks
    except:
        centre_guess1 = rad_pos
    
    # the initial guesses for the Gaussians
    # for 1: height, centre, width, offset
    guess1 = [-1., centre_guess1, width_guess]
        
    # make the array into the correct format
    data = np.array([range(len(points)), points]).T
    
    # the functions to be minimized
    errfunc1 = lambda p, xdata, ydata: (gaussian(xdata, *p) - ydata)
 
    optim1, success = optimize.leastsq(errfunc1, guess1[:], args=(data[:,0], data[:,1]))
    #optim1, success = curve_fit(gaussian, data[:,0], data[:,1], p0 = [height, centre_guess1, width_guess])

    
    return optim1, success


def rms_noise_signal(signal):
    """
    Find the rms of noise
    
    TODO: use non local means for the noise
    """
    rms_sq = [item**2 for item in signal]
    rms_noise = np.sqrt(np.sum(rms_sq) / len(rms_sq))

    return rms_noise

################################################################

def distance_3D(c1, c2):
    return np.sqrt( (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)


def find_contact_3D(centroids, radius, tol = 1):
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
            L = distance_3D(c1, c2)
            
#             if np.allclose(D, L, 0, 2):
            if abs(D - L) <= tol:
                touch_pt = ((c1[0] + c2[0]) / 2., (c1[1] + c2[1]) / 2., (c1[2] + c2[2]) / 2.) 
                print c1, " ", c2, "are in contact"
                print "touch point is ", touch_pt
                touch_pts.append(touch_pt)
                centres.append((c1, c2))
                
    return touch_pts, centres


def vector_3D(pt1, pt2, t):
    """
    Write the 3d line eqn in a parametric form
    (x,y,z) = (x1,y1,z1) - t * (x1-x2, y1-y2, z1-z2)
    x = x1 - (x1 - x2)*t
    y = y1 - (y1 - y2)*t
    z = z1 - (z1 - z2)*t
    """
    x1, y1, z1 = pt1
    x2, y2, z2 = pt2
    
    modulus = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    
    x = x1 - (x1 - x2) / modulus * t
    y = y1 - (y1 - y2) / modulus * t
    z = z1 - (z1 - z2) / modulus * t
    
    return [int(x), int(y), int(z)]


def vector_perpendicular_3D(pt1, pt2, which, Z, Sx):
    """
    Returns a vector S perpendicular to a line
    between pt1 and pt2 AND it lies in xy plane
    at height Z
    """

    x1, y1, z1 = pt1
    x2, y2, z2 = pt2
    
    # L is a vector between pt1 and pt2
    Lx = x1 - x2
    Ly = y1 - y2
    
    try:
        slope = -np.float(Lx) / Ly
    except:
        slope = 0
        
        
    if which == 1:
        Sy = Sx * slope + pt1[1]
        Sx = Sx + pt1[0]
        Sz = pt1[2]
        
    elif which == 2:
        Sy = Sx * slope + pt2[1]
        Sx = Sx + pt2[0]
        Sz = pt2[2]
        
    return [int(Sx), int(Sy), int(Sz + Z)]


def touch_lines_3D(pt1, pt2, image):
    """
    Goes along lines in the region between
    two points.
    """
    
    centre_dist = int(distance_3D(pt1, pt2) / 2.0)
    
    Xrange = range(-centre_dist, centre_dist)
    Zrange = range(-centre_dist, centre_dist)

    
    pixels = []
    
#     Xrange = range(-2, 2)
#     Zrange = range(0, 2)
    store_optim = 1000
    
    for Z in Zrange:
        for X in Xrange:
            
            # Draw a line parallel to the one through the 
            # point of contact at height Z
            P1 = vector_perpendicular_3D(pt1, pt2, 1, Z, X)
            P2 = vector_perpendicular_3D(pt1, pt2, 2, Z, X)
            
#             print "p1", P1
#             print "p2", P2
             
            line = []
            length = int(distance_3D(P1, P2))
            
            # go along that line
            for time in range(length):
                
                try:
                    x, y, z = vector_3D(P1, P2, time)
#                     print x, y, z
                    line.append(image[x, y, z])
                except:
                    continue
            
            
            if line:
                pixels.append(line)
                data = np.array([range(len(line)), line]).T
                
                optim1, suc = fit_gaussian_to_signal(line)
                
#                 # find the smallest stdev
#                 # Should be at touch point
#                 if abs(store_optim) > abs(optim1[2]):
#                     store_optim = abs(optim1[2])
#                     print "width ", abs(optim1[2]), "at ", P1, P2
#                     
#                 
#                     pl.plot(data[:,0], data[:,1], lw=5, c='g', label='measurement')
#                     pl.plot(data[:,0], gaussian(data[:,0], *optim1),
#                         lw=3, c='b', label='fit of 1 Gaussian')
#                     pl.legend(loc='best')
#                     pl.show()  
                
            if Z == 0:
                if X > -15:
                    print P1, P2
                    data = np.array([range(len(line)), line]).T
                
                    pl.plot(data[:,0], data[:,1], lw=5, c='g', label='measurement')
                    pl.plot(data[:,0], gaussian(data[:,0], *optim1),
                        lw=3, c='b', label='fit of 1 Gaussian')
                    pl.legend(loc='best')
                    pl.show()                
                
    return

# pt1 = (1, 1, 1)
# pt2 = (1, 5, 1)
#  
# touch_lines_3D(pt2, pt1)






