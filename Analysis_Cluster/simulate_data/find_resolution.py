import numpy as np
import pylab as pl
from scipy import optimize
from math import isnan
from scipy.ndimage.filters import gaussian_filter
from skimage import io
import os

#################################### Fitting fns ############################



def fit_gaussian(points, guess, weight, which):
    """
    Detects peaks
    First guess for the Gaussian is the firs maxima in the signal
    """
    # TODO: shift the other side of the peak if they are not on the same level
    filtered = use_filter(points, weight, which)
    data = np.array([range(len(filtered)), filtered]).T
    if guess == False:
        return [[0,0,0,0], data]
    else:
        # the functions to be minimized
        errfunc1 = lambda p, xdata, ydata: (gaussian(xdata, *p) - ydata)
     
        # TODO: PCOV SQRT GIVES THE ERRORS FOR THE STDEV
        p, pcov = optimize.leastsq(errfunc1, guess[:],
                                    args=(data[:,0], data[:,1]))
            
        return p, data


def gaussian(x, height, center, width, offset):
    return (height/np.sqrt(2*np.pi)*width) * np.exp(-(x - center)**2/(2*width**2)) + offset


def create_dir(directory):
    
    if not os.path.exists(directory):
        os.makedirs(directory)


def parameter_estimates_stats(points):
    """
    Obtain guesses from simple analysis
    """
    try:
        data = np.array([range(len(points)), points]).T
        centre_guess = np.argwhere(min(points) == points).T[0][0]
        height_guess = np.max(points) - abs(np.min(points)) 
        
        guess = [round(-abs(height_guess), 3), centre_guess, 10., round(abs(height_guess), 3)]
        return guess
    except:
        print "Not resolved"
        return False
    
    

def parameter_estimates_mad(signal, tol, value):
    """
    Apply MAD to extract the peak
    Take a small region around it and estimate the width
    centre, height etc.
    
    Then use this on the original image to fit
    and obtained a proper statistical measurement
    of resolution
    
    Also.. noise can be estimated by doing MAD, but
    not replacing outliers with 0
    """
    # Region of the Gaussian peak
    peak, indices = get_peak(signal, tol, value)
    
    if indices:
        sigma = len(indices)
        #print "indices", indices
        centre = np.median(indices) 
        height = np.max(signal) - abs(signal[int(centre)])
        
        guess = [round(-abs(height),3), centre, sigma, round(abs(height),3)]
        return guess
    else:
        print "Not resolved"
        return False
    


def fit_and_visualize(image, signals, coords, folder_name, touch_pt):
    """
    Take in an array containing all of the lines
    Take in the image along the slanted plane
    Take in the coordinates for each pixel
    
    Loop through the lines array and plot that line
    onto the image
    At the same time fit a gaussian for each line
    
    Plot each subplot and store it
    
    Do this for every Z value
    """
    import scipy

    max_int = np.max(image)
        
    for i in range(len(coords)):
        coord = coords[i]
        signal = signals[i]
        img_copy = image.copy()
        
        for j in range(len(coord)):
            x, y = coord[j]
            
            img_copy[x, y] = max_int
        #scipy.misc.imsave('outfile.jpg', img_copy)
        if signal:
            
            # PLOT THE IMAGE WITH THE LINE ON IT
            pl.subplot(2, 3, 1)        
            pl.imshow(img_copy)
            pl.gray()
            #pl.plot(touch_pt, touch_pt, 'bo')
       
            # DETERMINE Y LIMIT
            ymax = np.max(img_copy)
            ymin = np.min((img_copy))
            weight = 4
            tol = 4 # MAD median distance higher the better - gap is a super anomally
            data = np.array([range(len(signal)), signal]).T
            
            # No filter, bad guess
            pl.subplot(2, 3, 2)
            guess = parameter_estimates_stats(signal)
            print "Stats guess", guess
            param, unused = fit_gaussian(signal, guess, weight, 0)      
            pl.plot(data[:,0], data[:,1])
            pl.plot(data[:,0], gaussian(data[:,0], *param))
            pl.title("Stats guess / STD {0}".format(abs(round(param[2], 2))))
            pl.ylim(ymin,ymax)

            # No filter, MAD guess
            pl.subplot(2, 3, 3)  
            guess = parameter_estimates_mad(signal, 4, 1)
            print "MAD guess", guess
            param, unused = fit_gaussian(signal, guess, weight, 0)      
            pl.plot(data[:,0], data[:,1])
            pl.plot(data[:,0], gaussian(data[:,0], *param))
            pl.title("MAD guess / STD {0}".format(abs(round(param[2], 2))))
            pl.ylim(ymin,ymax)
            
            # MAD FILTER DATA
            pl.subplot(2, 3, 4) 
            mad_sig = noise_MAD(signal, tol)
            c = np.array([range(len(mad_sig)), mad_sig]).T       
            pl.plot(c[:,0], c[:,1])
            pl.title("Noise")
            pl.ylim(ymin,ymax)
            
            # MAD - SIGNAL DATA
            pl.subplot(2, 3, 5)
            peak, indices = get_peak(signal, tol, 10)
            c = np.array([range(len(peak)), peak]).T
            pl.plot(c[:,0], c[:,1])
            pl.title("Extracted peak")
            #pl.ylim(ymin,ymax)
            
            # MOVING AVERAGE FILTERED SIGNAL
            pl.subplot(2, 3, 6)
            mad_sig = thresh_MAD(signal, tol, 0)
            c = np.array([range(len(mad_sig)), mad_sig]).T
            pl.plot(c[:,0], c[:,1])
            pl.title("MAD")
            pl.ylim(ymin,ymax)
            
            pl.savefig(folder_name + 'result%i.png' % i)
            pl.close('all')
            
    return

######################## Signal analysis #########################

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


def get_peak(signal, tol, value):
    mad = MAD(signal)
    M = np.median(np.sort(signal))
    up_thresh = M + mad * tol
    bot_thresh = M - mad * tol
    
    only_peak = []
    
    for i in signal:
        if i <= up_thresh and i >= bot_thresh:
            only_peak.append(0)
        else:
#             peak_indices.append(np.argwhere(i == signal))
            only_peak.append(value)
  
    peak_indices = [index for index, item in enumerate(only_peak) if item == 1]  
    return only_peak, peak_indices


def noise_MAD(signal, tol):
    """
    Remove outliers using MAD
    """
    mad = MAD(signal)
    M = np.median(np.sort(signal))
    up_thresh = M + mad * tol
    bot_thresh = M - mad * tol
    
    clean_signal = []
    
    for i in signal:
        if i <= up_thresh and i >= bot_thresh:
            clean_signal.append(i)
    
    return clean_signal    


def thresh_MAD(signal, tol, value):
    """
    Remove outliers using MAD
    """
    mad = MAD(signal)
    M = np.median(np.sort(signal))
    up_thresh = M + mad * tol
    bot_thresh = M - mad * tol
    
    clean_signal = []
    
    for i in signal:
        if i <= up_thresh and i >= bot_thresh:
            clean_signal.append(i)
        else:
            clean_signal.append(value)
            
    return clean_signal


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / float(n)

def use_filter(signal, weight, which):
    """
    1 = gaussian
    2 = moving average
    3 = MAD
    4 = exp mov avg
    
    NOTES:
    gaussian is not the worst
    moving average seems to be really good!
    MAD destroys the main peak - bad! (try subtracting noise from signal???)
    wavelets dont even return a filtered signal

    """
    import pywt

    if which == 1:
        filtered = gaussian_filter(signal, weight)
        return filtered
    elif which == 2:
        filtered = moving_average(signal, weight)
        return filtered
    elif which == 3:
        filtered = thresh_MAD(signal)
        return filtered
    else:
        return signal
    
    

##################################################################

def modulus(pt):
    return np.sqrt(pt[0]**2 + pt[1]**2 + pt[2]**2)


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
    
    x = x1 + (x2 - x1) / modulus * t
    y = y1 + (y2 - y1) / modulus * t
    z = z1 + (z2 - z1) / modulus * t
    
    return [x, y, z]


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
        
    return [Sx, Sy, Sz + Z]


################## Points of contact ###############################

def find_contact_3D(centroids, radius, tol = 1.):
    """
    Input: Arrays of all the centroids and all radii
            tol defines the error tolerance between radii distance 
    
    Check all centre pairs and determine,
    based on their radii, if they are in contact
    or not
    """
    touch_pts = []
    centres = []
    radii = []
    
    N = len(centroids)
    for i in range(N - 1):
        for j in range(i + 1, N):
            
            c1 = centroids[i]
            c2 = centroids[j]
            r1 = radius[i]
            r2 = radius[j]
            
            D = r1 + r2
            L = distance_3D(c1, c2)
            print "distance between pts", L
            print "radii sum", D
            if abs(D - L) <= tol:
                
                touch_pt = vector_3D(c1, c2, r1)
                print c1, " ", c2, "are in contact"
                print "touch point is ", touch_pt
                touch_pts.append(touch_pt)
                centres.append((c1, c2))
                radii.append((r1, r2))
                
    return centres, touch_pts, radii   


########################### Crop the area #####################################

def crop_area(C1, C2, R1, size, name):
    """
    Get touch point
    Generate a pixel map of the box to be cropped
    Check at which pixels the line between centres starts and ends
    Start and end points will be used to loop through the box
    """
    touch_pt = vector_3D(C1, C2, R1)
    x, y, z = touch_pt
    x, y, z = (int(round(x,0)), int(round(y,0)), int(round(z,0)))
    dim = size * 2
    dim += 1

    area = np.zeros((dim, dim, dim))
    
    for i in np.arange(-size, size+1):
        
        # which slice to take
        zdim = z + i
        input_file = name % (zdim)    
        img = io.imread(input_file)
        
        # crop the slice and store
        area[:,:,i+size] = img[x - size:x + size + 1, y - size: y + size + 1]
    
    return area


def plot_images():
    """
    If centres and contact point lie in the same
    z plane then a slice of the image will show the situation.
    
    If they are at an angle the slicing is not trivial anymore
    """
    
    
    return 


def centres_shifted_to_box(C1, C2, size):
    """
    Find the projections of the centres on the box
    Also scale to the dimensions of the box
    
    These are needed for looping through that box
    """
    xv = C1[0] - C2[0]
    yv = C1[1] - C2[1]
    zv = C1[2] - C2[2]
    mod = modulus((xv, yv, zv))
    
    # vector pointing away from the touch point
    v = (xv/mod, yv/mod, zv/mod)
    
    # max distance within box
    possible_centres = []
    for t in range(4*size):
        C = (size - v[0] * t, size - v[1] * t, size - v[2] * t)
        possible_centres.append(C)
        if C[0] < 0 or C[1] < 0 or C[2] < 0:
            mod_C1 = possible_centres[-2]
            break
    else:
        for t in range(4*size):
            C = (size + v[0] * t, size + v[1] * t, size + v[2] * t)
            possible_centres.append(C)
            if C[0] < 0 or C[1] < 0 or C[2] < 0:
                mod_C1 = possible_centres[-2]
                break
    
    # Once one centre is know from trig get another
    # find distance between new centre and touch pt
    dist = distance_3D(mod_C1, (size, size , size)) * 2
    
    mod_C2 = vector_3D(mod_C1, (size, size, size), dist)
    
    return [mod_C1, tuple(mod_C2)]    


def touch_lines_3D(pt1, pt2, image, sampling, touch_pt, folder_name):
    """
    Goes along lines in the region between
    two points.
    Used for obtaining the widths of the gaussian fitted
    to the gap between spheres
    """    
    centre_dist = round(distance_3D(pt1, pt2) / 2.0, 0)
    
    Xrange = np.arange(-centre_dist, centre_dist + 1)
#     Zrange = np.linspace(-centre_dist, centre_dist + 1)
    Zrange = np.arange(0.,1.)  # Take only the centre
    
    for Z in Zrange:
        # Create an array to store the slanted image slices
        # used for plotting
        plot_image = np.zeros((int(distance_3D(pt1, pt2)), int(distance_3D(pt1, pt2))), dtype=image.dtype)
        lines = []
        coords = []
        
        for X in Xrange:
            # Draw a line parallel to the one through the 
            # point of contact at height Z
            P1 = vector_perpendicular_3D(pt1, pt2, 1, Z, X)
            P2 = vector_perpendicular_3D(pt1, pt2, 2, Z, X)
            
            line = []
            coord = []
            length = distance_3D(P1, P2)

            # go along that line
            for time in np.linspace(0., length + 1, length*sampling):
                try:
                    # line coordinates going through the gap
                    x, y, z = vector_3D(P1, P2, time)
                    x, y, z = (int(round(x,0)), int(round(y,0)), int(round(z,0)))
                    
                    pixel_value = image[x, y, z]
                    plot_image[int(round(X + touch_pt,0)), int(round(time,0))] = pixel_value
                    
                    line.append(pixel_value)
                    coord.append((int(round(X + touch_pt,0)), int(round(time,0))))
                except:
                    continue
            
            lines.append(line)
            coords.append(coord)
        
        folder_name = folder_name + "plots_%i/" % (Z + touch_pt)
        create_dir(folder_name)
        fit_and_visualize(plot_image, lines, coords, folder_name, touch_pt)

        
    return