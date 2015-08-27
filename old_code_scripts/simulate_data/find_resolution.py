import numpy as np
import pylab as pl
from scipy import optimize
from math import isnan
from scipy.ndimage.filters import gaussian_filter, median_filter,\
    gaussian_filter1d
from skimage import io
import os
from skimage.filter import denoise_tv_chambolle
import fit_data
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
        try:
            p, pcov = optimize.leastsq(errfunc1, guess[:],
                                    args=(data[:,0], data[:,1]))
        except:
            p, pcov = [[0,0,0,0],0]
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
        
        prob = gaussian_filter(points, 6)
#         mu   = data[:,0].dot(prob)               # mean value
#         mom2 = np.power(data[:,0], 2).dot(prob)  # 2nd moment
#         var  = mom2 - mu**2               # variance
        mu = prob.dot(data[:, 0])  # mean or location param.
        sigma = np.sqrt(prob.dot(data[:, 0] ** 2) - mu ** 2)
        sigma_guess = sigma
        
        guess = [round(-abs(height_guess), 3), centre_guess, sigma_guess, round(abs(height_guess), 3)]
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
    


def fit_and_visualize(image, signals, coords, folder_name):
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
        new_signal = []
        
        for j in range(len(coord)):
            x, y = coord[j]
            new_signal.append(image[x, y])
            img_copy[x, y] = max_int
            
        #scipy.misc.imsave('outfile.jpg', img_copy)
        if signal:
            signal = new_signal
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
            # TODO: for poor contrast increase the tolerance
            data = np.array([range(len(signal)), signal]).T
            
            # STATS GUESS
            pl.subplot(2, 3, 2)
            guess = parameter_estimates_stats(signal)
            print "Stats guess", guess
            param, unused = fit_gaussian(signal, guess, weight, 0)      
            pl.plot(data[:,0], data[:,1])
            pl.plot(data[:,0], gaussian(data[:,0], *param))
            pl.title("Stats / STD {0}".format(abs(round(param[2], 2))))
            pl.ylim(ymin,ymax)
            #pl.ylim(0, 1.2)
            
            
            # No filter, MAD guess
            pl.subplot(2, 3, 3)  
            guess = parameter_estimates_mad(signal, 4, 1)
            print "MAD guess", guess
            param, unused = fit_gaussian(signal, guess, weight, 0)      
            pl.plot(data[:,0], data[:,1])
            pl.plot(data[:,0], gaussian(data[:,0], *param))
            pl.title("MAD / STD {0}".format(abs(round(param[2], 2))))
            pl.ylim(ymin,ymax)
            #pl.ylim(0, 1.2)
            
            # MAD FILTER DATA
            pl.subplot(2, 3, 4) 
            guess = parameter_estimates_stats(signal)
            X, best_fit, chi = fit_data.step(signal, guess)
            pl.plot(data[:,0], data[:,1])
            pl.plot(X, best_fit)
            pl.title("Step")
            pl.ylim(ymin,ymax)
            #pl.ylim(0, 1.2)
            
            pl.subplot(2, 3, 5)
            guess = parameter_estimates_stats(signal)
            X, best_fit, chi = fit_data.gauss_step_const(signal, guess)
            pl.plot(data[:,0], data[:,1])
            pl.plot(X, best_fit)
            pl.title("Contrast fit")
            pl.ylim(ymin,ymax)
            #pl.ylim(0, 1.2)
            
            pl.subplot(2, 3, 6)
            guess = parameter_estimates_stats(signal)
            X, best_fit = fit_data.minimized_residuals(signal, guess)
            pl.plot(data[:,0], data[:,1])
            pl.plot(X, best_fit)
            pl.title("Minimum residual" )
            pl.ylim(ymin,ymax)
            #pl.ylim(0, 1.2)
            
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


def equalize(im,nbr_bins=256):
    """histogram equalization function
    it increases contrast and makes
    all pixel values to occur
    as comonly as other pixel values do"""
    
    #returns the frequency of all the 
    imhist,bins = np.histogram(im.flatten(), nbr_bins)#,density=True)
    #cdf of the histogram
    cdf = imhist.cumsum()
    #cdf[-1] gives the largest sum from the cdf i.e. largest probability. 
    #dividing by it produces probabilitie from 0 to 1
    #multiplying by 255 gives cdf from 0 to 255
    cdf = cdf / cdf[-1]
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    
    return im2.reshape(im.shape), bins   


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

    v = ((pt2[0] - pt1[0]), (pt2[1] - pt1[1]), (pt2[2] - pt1[2]))
    #v =  project_onto_plane(v, (0,0,1))
    if which == 1:
        Sx, Sy = (pt1[0] - v[1]/ np.sqrt(v[0]**2 + v[1]**2) * Sx, pt1[1] + v[0]/ np.sqrt(v[0]**2 + v[1]**2) * Sx)
        Sz = pt1[2]
    elif which == 2:
        Sx, Sy = (pt2[0] - v[1]/ np.sqrt(v[0]**2 + v[1]**2) * Sx, pt2[1] + v[0]/ np.sqrt(v[0]**2 + v[1]**2) * Sx)
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


def get_slices(P1, P2, name):
    """
    Get slices for analysis
    """
    zstart = np.min([P1[2], P2[2]])
    zend  = np.max([P1[2], P2[2]])
    
    height = zend - zstart
    
    slices = {}
    
    for h in range(zstart, zend + 1):
        input_file = name % (h)    
        img = io.imread(input_file)
        #eql = denoise_tv_chambolle(img, weight = 0.02)
        
        slices[h] = img
    
    return slices

def touch_lines_3D(pt1, pt2, sampling, folder_name, name):
    """
    Goes along lines in the region between
    two points.
    Used for obtaining the widths of the gaussian fitted
    to the gap between spheres
    """    
    
    centre_dist = round(distance_3D(pt1, pt2), 0)
    Xrange = np.arange(-centre_dist / 3., centre_dist / 3. + 1)
#     Zrange = np.linspace(-centre_dist, centre_dist + 1)
    Zrange = np.arange(0.,1.)  # Take only the centre
    
    for Z in Zrange:
        # Create an array to store the slanted image slices
        # used for plotting
        plot_image = np.zeros((centre_dist * 2. / 3., centre_dist))
        lines = []
        coords = []
        slices = get_slices(pt1, pt2, name)
        
        for X in Xrange:
            # Draw a line parallel to the one through the 
            # point of contact at height Z
            P1 = vector_perpendicular_3D(pt1, pt2, 1, Z, X)
            P2 = vector_perpendicular_3D(pt1, pt2, 2, Z, X)
            line = []
            coord = []
            length = distance_3D(P1, P2)
            
            # go along that line
            for time in np.linspace(0, length + 1, length*sampling):
                try:
                    # line coordinates going through the gap
                    x, y, z = vector_3D(P1, P2, time)
                    x, y, z = (int(round(x,0)), int(round(y,0)), int(round(z,0)))
                    
                    pixel_value = slices[z][x,y]
                    plot_image[X + centre_dist / 3., time] = pixel_value
                    
                    line.append(pixel_value)
                    coord.append((X + centre_dist / 3., time))
                except:
                    continue
            
            lines.append(line)
            coords.append(coord)
        
        folder_name = folder_name + "plots_%i/" % Z
        create_dir(folder_name)
        
        # TODO : change the equalization or median filtering
        #plot_image = median_filter(plot_image, 3)
        #plot_image, bins = equalize(plot_image, 2)
        #plot_image, bins = equalize(plot_image, 2)
        
        # bins hold [background, sphere1, sphere2]
        
        fit_and_visualize(plot_image, lines, coords, folder_name)

    return