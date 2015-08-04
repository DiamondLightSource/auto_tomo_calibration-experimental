import numpy as np
import pylab as pl
from scipy import optimize
from math import isnan


def distance_3D(c1, c2):
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)


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
    
    modulus = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    
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


def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset


def fit_gaussian_to_signal(points):
    """
    Detects peaks
    First guess for the Gaussian is the firs maxima in the signa;
    """
            
    # make the array into the correct format
    data = np.array([range(len(points)), points]).T
    
    # find the initial guesses
    prob = data[:, 1] / data[:, 1].sum()  # probabilities
    mu = prob.dot(data[:, 0])  # mean or location param.
    sigma = np.sqrt(prob.dot(data[:, 0] ** 2) - mu ** 2)
    
    if isnan(sigma):
        
        print "fitting failed - stop"
        return False, False
    else:
        try:
            centre_guess = mu
        except:
            centre_guess = len(points) / 2.
        
        try:
            height_guess = np.argwhere(points == np.min(points))[0][0]
        except:
            height_guess = 1.
                
        sigma_guess = sigma
        height_guess = np.argwhere(points == np.min(points))[0][0]
        
        # the initial guesses for the Gaussians
        guess1 = [-abs(height_guess), centre_guess, sigma_guess, abs(height_guess)]
    
        
        # the functions to be minimized
        errfunc1 = lambda p, xdata, ydata: (gaussian(xdata, *p) - ydata)
     
        p, success = optimize.leastsq(errfunc1, guess1[:],
                                    args=(data[:,0], data[:,1]))
        
        return p, sigma


################################################################
def MAD(signal):
    """
    Calculate median absolute deviation
    """
    # constant linked to data normality
    b = 1.4826
    
    # median of the data
    M = np.median(np.sort(signal))
    
    mad = b * np.median(np.sort(abs(signal - M)))
    
    return mad


def thresh_MAD(signal):
    """
    Remove outliers using MAD
    """
    mad = MAD(signal)
    M = np.median(np.sort(signal))
    up_thresh = M + mad
    bot_thresh = M - mad
    
    clean_signal = []
    
    for i in signal:
        if i <= up_thresh and i >= bot_thresh:
            clean_signal.append(i)
    
    return clean_signal


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
            
            if abs(D - L) <= tol:
                # TODO: Get the touch point - not midpoint
                
                #touch_pt = ((c1[0] + c2[0]) / 2., (c1[1] + c2[1]) / 2., (c1[2] + c2[2]) / 2.) 
                print c1, " ", c2, "are in contact"
                #print "touch point is ", touch_pt
                #touch_pts.append(touch_pt)
                centres.append((c1, c2))
                
    return centres   


def touch_lines_3D(pt1, pt2, image):
    """
    Goes along lines in the region between
    two points.
    """
    
    centre_dist = int(distance_3D(pt1, pt2) / 4.0)
    
    Xrange = range(-centre_dist, centre_dist)
    Zrange = range(-centre_dist, centre_dist)

    
    min_widths = []
    mean_widths = []
    median_widths = []
    total_widths = []
    
    for Z in Zrange:

        widths = []
        
        for X in Xrange:
            
            # Draw a line parallel to the one through the 
            # point of contact at height Z
            P1 = vector_perpendicular_3D(pt1, pt2, 1, Z, X)
            P2 = vector_perpendicular_3D(pt1, pt2, 2, Z, X)
             
            line = []
            length = int(distance_3D(P1, P2))
            
            # go along that line
            for time in range(length):
                try:
                    x, y, z = vector_3D(P1, P2, time)
                    line.append(image[x, y, z])
                except:
                    continue
            
            # if it is not empty
            if line:

                # Best Gaussian fit
                parameters, sigma = fit_gaussian_to_signal(line)
                
                # find the smallest standard deviation
                # Should be at near the touch point
                if sigma != False:

                    widths.append(abs(parameters[2]))
                    total_widths.append(abs(parameters[2]))

        # clean all the outliers
        cleaned_widths = thresh_MAD(thresh_MAD(widths))
        cleaned_total = thresh_MAD(thresh_MAD(total_widths))
        
        # get the minimum, mean and the median width
        # for each slice
        min_widths.append(np.min(cleaned_widths))
        mean_widths.append(np.mean(cleaned_widths))
        median_widths.append(np.median(np.sort(cleaned_widths)))
    
    print ""
    print "GETS A SIGNAL ALONG A LINE AROUND THE POINT OF CONTACT"
    print "STORES THE GAUSSIAN WIDTH OF THAT SIGNAL"
    print "DOES THIS FOR THE WHOLE PLANE"
    print "AND THEN TAKES MINIMUM / MEAN / MEDIAN WIDTH OF THE WHOLE SLICE"
    print "THIS IS REPEATED FOR DIFFERENT HEIGHTS AND AN ARRAY OF"
    print "MINIMUM / MEAN / MEDIAN WIDTHS IS CREATED"
    print ""
    print "ARRAY WITH MIN WIDTHS OF EACH SLICE"
    print "min", np.min(min_widths)
    print "mean", np.mean(min_widths)
    print "median", np.median(np.sort(min_widths))
    print ""
    
    print "ARRAY WITH MEAN WIDTHS OF EACH SLICE"
    print "min sigma", np.min(mean_widths)
    print "mean sigma", np.mean(mean_widths)
    print "median sigma", np.median(np.sort(mean_widths))
    print ""
    
    print "ARRAY WITH MEDIAN WIDTHS OF EACH SLICE"
    print "min sigma", np.min(median_widths)
    print "mean sigma", np.mean(median_widths)
    print "median sigma", np.median(np.sort(median_widths))
    print ""

    print "THESE ARE ALL OF THE WIDTHS OF ALL THE SIGNALS"
    print "AT EVERY HEIGHT"
    print ""
    print "ARRAY WITH WIDTHS OF ALL THE AREA"
    print "min sigma", np.min(cleaned_total)
    print "mean sigma", np.mean(cleaned_total)
    print "median sigma", np.median(np.sort(cleaned_total))
    print ""


    return np.mean(cleaned_total)
