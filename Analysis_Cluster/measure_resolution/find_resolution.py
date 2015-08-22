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
from scipy import misc
from math import ceil
#################################### Fitting fns ############################



def sgolay2d( z, window_size, order, derivative=None):
    from scipy import signal
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return signal.fftconvolve(Z, -r, mode='valid'), signal.fftconvolve(Z, -c, mode='valid')



def remove_noise(signal, r1, r2, Y, C):
    from scipy.fftpack import fftshift, fft
    from scipy.signal import savgol_filter
#     mtf_fn = fftshift(abs(fft(signal)))
    h = C - Y
    
    d1 = np.sqrt(r1**2 - h**2)
    dist = int(dist_between_spheres(r1, r2, Y, C))
    
    if dist % 2 == 0:
        dist += 1
        
    if dist < 5:
        dist = 5
        
    filtered = savgol_filter(signal, int(dist), 4)
    return filtered



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


def parameter_estimates_stats(points, distance):
    """
    Obtain guesses from simple analysis
    """
    try:
        
        data = np.array([range(len(points)), points]).T
        xs = data[:,0]
        ys = data[:,1]
        centre_guess = np.argwhere(min(points) == points).T[0][0]
#         centre_guess = len(points) / 2.
        height_guess = np.max(points) - abs(np.min(points))
        
        guess = [round(-abs(height_guess), 3), centre_guess, distance + 2, round(abs(height_guess), 3)]
        return guess
    except:
        print "Not resolved"
        return False
 
    
def add_noise(np_image, amount):
    """
    Adds random noise to the image
    """
    noise = np.random.randn(np_image.shape[0],np_image.shape[1])
    norm_noise = noise/np.max(noise)
    np_image = np_image + norm_noise*np.max(np_image)*amount
    
    return np_image   


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
    


def fit_and_visualize(image, folder_name, r1, r2):
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
    
    misc.imsave(folder_name + "touch_img.png", image)
    misc.imsave(folder_name + "touch_img.tif", image)
#     image = io.imread("/dls/tmp/jjl36382/resolution/plots1/7/plots_0/touch_img.tif")
    denoised = image 
    contrast_left = measure_contrast_left(denoised)
    contrast_right = measure_contrast_right(denoised)
    
    low_freq_left = (contrast_left - np.min(denoised[0,:])) /\
                    (contrast_left + np.min(denoised[0,:]))
    low_freq_right = (contrast_right - np.min(denoised[0,:])) /\
                     (contrast_right + np.min(denoised[0,:]))
    
    gap = []
    mtf_cleft = []
    mtf_cright = []
     
    for i in np.arange(0, image.shape[0]/2., 0.05):
         
        Xdata = []
        Ydata = []
        gapX = []
        gapY = []
        
        distance = dist_between_spheres(r1, r2, i, image.shape[0]/2.)
        
        signal = [pixel for pixel in denoised[int(round(i,0)),:]]
        for j in np.arange(0, image.shape[1], 0.1):
             
            Xdata.append(j)
            Ydata.append(i)
            if image.shape[1]/2. + distance/2. >= j >= image.shape[1]/2. - distance/2.:
                gapX.append(j)
                gapY.append(i)
             
#         if signal:
#             
#             # PLOT THE IMAGE WITH THE LINE ON IT
#             if i > 50:
#                 pl.imshow(denoised)
#                 pl.plot(Xdata, Ydata)
#                 pl.plot(gapX, gapY)
#                 pl.gray()
#                 pl.axis('off')
#                 pl.savefig(folder_name + 'result{0}.png'.format(i))
#                 pl.close('all')
            
#             # DETERMINE Y LIMIT
#             ymax = np.max(denoised)
#             ymin = np.min(denoised)
#                 
#             # MLMFIT
#             data = np.array([range(len(filtered_signal)), filtered_signal]).T
#              
#             pl.subplot(1, 2, 2)
#             distance = dist_between_spheres(r1, r2, i, image.shape[0]/2.)
#             guess = parameter_estimates_stats(filtered_signal, distance)
#             X, best_fit, cent, fwhm = fit_data.GaussConst(filtered_signal, guess)
#             pl.plot(data[:,0], data[:,1])
#             pl.plot(X, best_fit)
#             pl.title("FWHM {0}".format(round(fwhm,2)))
#             pl.ylim(ymin,ymax)
#              
# #             pl.savefig("./" + 'result%i.png' % i)
#             pl.savefig(folder_name + 'result%i.png' % i)
#             pl.close('all')
# 
#             if fwhm < 8:
#                 mtf = 100 * modulation(np.min(signal), contrast_left, distance) / low_freq_left
#                 # bellow this limit the spheres are unresolved
#                 # and the width gets distorted - drop this data
#                 if mtf > 9:
#                     mtf_cleft.append(mtf)
#                     mtf_fwhm_left.append(fwhm)
#      
#                  
#                 mtf = 100 * modulation(signal[int(cent)], contrast_right, distance) / low_freq_right
#                 # bellow this limit the spheres are unresolved
#                 # and the width gets distorted - drop this data
#                 if mtf > 9:
#                     mtf_cright.append(mtf)
#                     mtf_fwhm_right.append(fwhm)

        distance = dist_between_spheres(r1, r2, i, image.shape[0] / 2.)
        gap.append(distance)
               
        mtf = 100 * modulation(np.min(signal), contrast_left, distance) / low_freq_left
        # bellow this limit the spheres are unresolved
        # and the width gets distorted - drop this data
        mtf_cleft.append(mtf)
        
        mtf = 100 * modulation(np.min(signal), contrast_right, distance) / low_freq_right
        # bellow this limit the spheres are unresolved
        # and the width gets distorted - drop this data
        mtf_cright.append(mtf)

    ############# LEFT SPHERE #########################
    best_fit, limit = fit_data.MTF(gap, mtf_cleft)
    pl.gca().invert_xaxis()
    pl.plot(best_fit, mtf_cleft, label = "best fit")

    mtf_resolutionX = [item for item in gap if item > limit]
    mtf_resolutionY = [item for item in mtf_cleft if item < 9]

    pl.plot(gap, mtf_cleft, 'r,', label="left sphere")
    pl.plot(mtf_resolutionX, np.repeat(9, len(mtf_resolutionX)), 'y')
    pl.plot(np.repeat(limit, len(mtf_resolutionY)), mtf_resolutionY, 'y')
    pl.legend()
    pl.title("Gap width at 9% (Rayleigh diffraction limit) is {0}".format(limit))
    pl.xlabel("Width")
    pl.ylabel("MTF %")
    pl.xlim(np.max(gap), 0)
    pl.ylim(0, 110)
    pl.savefig(folder_name + 'mtf_left.png')
    pl.tight_layout()

    pl.close('all')
    
    ############### RIGHT SPHERE #####################
    best_fit, limit = fit_data.MTF(gap, mtf_cright)
    pl.gca().invert_xaxis()
    pl.plot(best_fit, mtf_cleft, label = "best fit")

    mtf_resolutionX = [item for item in gap if item > limit]
    mtf_resolutionY = [item for item in mtf_cright if item < 9]

    pl.plot(gap, mtf_cright, 'b,', label="right sphere")
    pl.plot(mtf_resolutionX, np.repeat(9, len(mtf_resolutionX)), 'y')
    pl.plot(np.repeat(limit, len(mtf_resolutionY)), mtf_resolutionY, 'y')
    pl.legend()
    pl.title("Gap width at 9% (Rayleigh diffraction limit) is {0}".format(limit))
    pl.xlabel("Width")
    pl.ylabel("MTF %")
    pl.xlim(np.max(gap), 0)
    pl.ylim(0, 110)
    pl.savefig(folder_name + 'mtf_right.png')
    
    return

def gap(r1, r2, Y, C,):
    h = C - Y
    
    d1 = np.sqrt(r1**2 - h**2)
    
    return d1

def dist_between_spheres(r1, r2, Y, C):
    
    h = C - Y
    
    d1 = np.sqrt(r1**2 - h**2)
    d2 = np.sqrt(r2**2 - h**2)

    dist = r1 - d1 + r2 - d2
    
    return dist


def modulation(minima, contrast, distance):
        
    numerator = contrast - minima
    denominator = contrast + minima
    
    return numerator / denominator

def crop_peak(signal, r1, r2, Y, C):
    
    h = C - Y
    
    d1 = np.sqrt(r1**2 - h**2)
    dist = dist_between_spheres(r1, r2, Y, C)
    
    d1 = len(signal) / 2. - dist / 1.5
    
    
    print len(signal)
    print d1
    print d1 + dist
    val = np.max(signal)
    
    for i in range(len(signal)):
        if i < (d1) or i > (d1 + dist):
            signal[i] = val
    
    return signal


######################## Signal analysis #########################
def measure_contrast_left(image):
    """
    Measure contrast of the sphere being
    analyse which is on the left side of the image
    """
    pixels = []

    for j in range(image.shape[0]):
        pixels.append(image[j, 1])
#         image[j, 5] = 0

    return np.mean(pixels) 


def measure_contrast_right(image):
    """
    Measure contrast of the sphere being
    analyse which is on the left side of the image
    """
    pixels = []

    for j in range(image.shape[0]):
        pixels.append(image[j, image.shape[1] - 1])
#         image[j, image.shape[1]-5] = 0

    return np.mean(pixels) 

    return


def measure_contrast(signal):
        
    len_left = int(round(len(signal) / 3., 0))
    len_right = int(round(len(signal) * 2 / 3., 0))
    
    values_left = [pixel for pixel in signal[0:len_left]]
    values_right = [pixel for pixel in signal[len_right:len(signal)]]
            
#     CYleft = [i for coord in range(0, len_left)]
#     CXleft = [coord for coord in range(0, len_left)]
#     
#     CYright = [i for coord in range(len_right,image.shape[1])]
#     CXright = [coord for coord in range(len_right,image.shape[1])]
#             
#     pl.plot(CXleft, CYleft, '*')
#     pl.plot(CXright, CYright, '+')

    left = np.mean(values_left)
    right = np.mean(values_right)
    
    contrast_L = np.min(signal) / left
    contrast_R = np.min(signal) / right
    
    return contrast_L, contrast_R
    

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
    
    noise = []
    
    for i in signal:
        if i >= up_thresh and i <= bot_thresh:
            noise.append(i)
    
    return noise


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
            print abs(D - L)
            if abs(D - L) <= tol:
                
                touch_pt = vector_3D(c1, c2, r1)
                touch_pts.append(touch_pt)
                centres.append((c1, c2))
                radii.append((r1, r2))
                
    return centres, touch_pts, radii


def get_slice(P1, P2, name, sampling, Z):
    """
    Get slice through centre for analysis
    """
    
    centre_dist = distance_3D(P1, P2)
    plot_img = np.zeros((ceil(centre_dist / 4. + 1), centre_dist / 5. + 2 ))
    
    Xrange = np.arange(-centre_dist / 8., centre_dist / 8. + 1)
    
    for time in np.linspace(centre_dist*0.4, centre_dist*0.6 + 1,
                            centre_dist / 2.* sampling):
        # Go up along the line
        new_pt = vector_3D(P1, P2, time)
        old_pt = vector_3D(P1, P2, time - centre_dist / 2.* sampling)
        
        # If this is not the first iteration
        if time == centre_dist*0.4:
            input_file = name % int(round(new_pt[2] + Z, 0))
            img = io.imread(input_file)
            
        # check if the previous slice is the same as the next
        # dont load it again if it is
        if int(round(new_pt[2] + Z, 0)) != int(round(old_pt[2] + Z, 0)):
            
            input_file = name % int(round(new_pt[2] + Z, 0))
            img = io.imread(input_file)
            
            for X in Xrange:
        
                # Get along the X direction for every height
                perp = vector_perpendicular_3D(new_pt, P2, 1, 0, X)
                
                pixel_value = img[perp[0], perp[1]]
                
                time_mod = time - centre_dist * 0.4
                plot_img[X + centre_dist / 8., time_mod] = pixel_value
        else:
            for X in Xrange:
        
                # Get along the X direction for every height
                perp = vector_perpendicular_3D(new_pt, P2, 1, 0, X)
                
                pixel_value = img[perp[0], perp[1]]
                
                time_mod = time - centre_dist * 0.4
                plot_img[X + centre_dist / 8., time_mod] = pixel_value
   
    return plot_img


def touch_lines_3D(pt1, pt2, sampling, folder_name, name, r1, r2):
    """
    Goes along lines in the region between
    two points.
    Used for obtaining the widths of the gaussian fitted
    to the gap between spheres
    """
    
    Zrange = np.arange(0.,1.)  # Take only the centre
    
    for Z in Zrange:
        # Create an array to store the slanted image slices
        # used for plotting
        slice = get_slice(pt1, pt2, name, sampling, Z)
        
        folder_name = folder_name + "plots_%i/" % Z
        create_dir(folder_name)
        
        fit_and_visualize(slice, folder_name, r1, r2)

    return

# # fit_and_visualize(1,1,382,382)
# p1 = ( 0.2 * 1280 + 1280, -0.2 * 1280 + 1280,1280.)
# p2 = ( 0.2 * 1280 + 1280, 0.2 * 1280 + 1280, 1280.)
# # print p1, p2
# # 
# touch_lines_3D(p2, p1, 2, "./", "/dls/tmp/jjl36382/resolution1/reconstruction/testdata1/image_%05i.tif", 0.2 * 1280, 0.2 * 1280)