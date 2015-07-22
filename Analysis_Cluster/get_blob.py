import pylab as pl
import numpy as np

import math
from math import pi, log

from scipy import optimize
from scipy.signal import argrelextrema, find_peaks_cwt
from scipy.integrate import simps
from scipy import fft, ifft
from scipy.optimize import curve_fit, leastsq
from peak_detect import *


def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset


def gaussian_no_offset(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset
        
    
def two_gaussians(x, h1, c1, w1, h2, c2, w2, offset):
    return (gaussian(x, h1, c1, w1, offset) + 
            gaussian(x, h2, c2, w2, offset) + offset)


def modulation_transfer_function(lsf):
    """
    Takes in an array of gaussians from a slice
    of the sphere and calculates the fft of each line spread function.
    This is the MTF of that point. Adding the areas bellow the MTF,
    a number corresponding to relative image quality is obtained 
    """
    # normalize
    lsf_norm = []
    lsf_areas = []
    for item in lsf:
        try:
            lsf_norm.append(item / simps(item))
            lsf_areas.append(simps(item))
#             lsf_norm.append(item / max(item))
        except:
            pass
    
    lsf_avg_area = np.mean(lsf_areas)
    
#     # get the fft of the ideal system i.e.
#     # lsf is a delta function
#     centres = [int(len(centre) / 2.0) for centre in lsf_norm]
#     empty_lists = [np.zeros(len(item)) for item in lsf_norm]
#     
#     # get the ABSOLUTE fft value of each line spread function
#     measured_fft = [np.fft.fftshift(np.fft.fft(item[centres[i]-10:centres[i]+10])) for i, item in enumerate(lsf_norm)]
# 
#     
#     dirac_deltas = []
#     for i in range(len(empty_lists)):
#         ideal_lsf = empty_lists[i]
#         ideal_lsf[centres[i]] = lsf_norm[i][centres[i]]
#         dirac_deltas.append(ideal_lsf)
#      
#     ideal_fft = [np.fft.fftshift(np.fft.fft(item)) for item in dirac_deltas]
#      
#     # Object = Image * Impulse response
#     # F[obj] = F[Img] . F[Impulse] = F[img] . MTF
#     # MTF = F[obj] / F[img]
#     mtfs = []
#     for i in range(len(ideal_fft)):
# #         mtf = np.asarray(np.divide(measured_fft[i], ideal_fft[i]))
#         mtf = abs(measured_fft[i])
#         #normalize
# #         mtf = mtf / simps(mtf)
# #         mtf = mtf / max(mtf)
#         mtfs.append(mtf)
#         
# #     nyquist = int(len(mtf) / 4.)
# #     pl.plot(mtfs[0][centres[0]:centres[0] + nyquist])
#     pl.plot(mtfs[0])
#     pl.title("Mtf using division; centered")
#     pl.show()
# #     get the impulse response function
# #     impulse_response = [np.fft.ifft(np.fft.ifftshift(item)) for item in measured_fft]
# #     # center it at zero
# #     pl.plot(impulse_response[0][centres[0]:])
# #     pl.title("Impulse response; centered")
# #     pl.show()
#     
#     pl.plot(lsf_norm[0])
#     pl.title("Original normalized LSF")
#     pl.show()
# 
#     
#     # get a single average value used for comparison of images
#     #print "mean mtf area", np.mean(mtfs)
    
    return lsf_avg_area





def rms_noise_image(image):
    """
    Find the rms of noise
    
    TODO: use non local means for the noise
    """
    
    Xc = int(image.shape[0] / 2.)
    Yc = int(image.shape[1] / 2.)
    
    # Denoise the image
    
    # Get values of pixels according to an increasing radius for a same angle
    # Get the radius to be at most half the image since the circle can't be bigger
    R = image.shape[0] / 4.
    
    
    
    # Simple trig identities
    # R is the max value that we can reach
    delta_x = R * math.sin(0)
    delta_y = R * math.cos(0)
    points = []
    # Go from 0 to 1.001 in steps of 0.001
    for alpha in np.arange(0, 1.001, 0.001):
            # Xc and Yc are the positions from the center
            # points stores all the points from the center going along the radius
            points.append(image[int(Xc + alpha * delta_x), int(Yc + alpha * delta_y)])

    
    # peak to peak value
    ptp = np.max(points) - np.min(points)
    
    rms_sq = [item**2 for item in points]
    rms_noise = np.sqrt(np.sum(rms_sq) / len(rms_sq))

    return rms_noise


def rms_noise_signal(signal):
    """
    Find the rms of noise
    
    TODO: use non local means for the noise
    """
    rms_sq = [item**2 for item in signal]
    rms_noise = np.sqrt(np.sum(rms_sq) / len(rms_sq))

    return rms_noise


def find_contact(points, noiz):
    """
    Finds the max intensity along the line
    Compares it with the background intensity
    If it is within the RMS of noise, then
    that point indicates an edge
    """
    
    max_int = np.max(points)

    if max_int >= np.mean(noiz) + 3 * np.std(noiz):
        return None
    else:
        return 1


def fit_gaussian_to_signal(points, sigma, take_abs, rad_pos = 0., height = 0., width_guess = 10.):
    """
                     FAIL PROOF DETECOTR (?)
     if the margins in the selector are changed then
     change the 1.3 factor
    """
    
#     rms_noise5 = rms_noise_signal(points)
    peaks = peakdetect(points / np.max(points),
                       lookahead = 15 * sigma)#, delta = rms_noise5 / np.max(points))
    
    try:
        centre_guess1 = peaks[0][0][0]
    except:
        centre_guess1 = rad_pos
    
    
    # the initial guesses for the Gaussians
    # for 1: height, centre, width, offset
    guess1 = [height, centre_guess1, width_guess, 0.]
        
    # make the array into the correct format
    data = np.array([range(len(points)), points]).T
    
    # the functions to be minimized
    errfunc1 = lambda p, xdata, ydata: (gaussian(xdata, *p) - ydata)
 
    optim1, success2 = optimize.leastsq(errfunc1, guess1[:], args=(data[:,0], data[:,1]))
    
    if take_abs:
        return abs(optim1)
    else:
        return optim1

def is_outlier(points, thresh=3):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh


def get_touch_point(intensity, sigma):
    
    rms_noise5 = rms_noise_signal(intensity)
    peaks_max, peaks_min = peakdetect(intensity,
                       lookahead = 15 * sigma, delta = rms_noise5)
    
    for i in peaks_min:
        
        print i
        centre_guess = i[0]
        height = 0
        # the initial guesses for the Gaussians
        # for 1: height, centre, width, offset
        guess1 = [height, centre_guess, 10, 0.]
            
        # make the array into the correct format
        data = np.array([range(len(intensity)), intensity]).T
        
        # the functions to be minimized
        errfunc1 = lambda p, xdata, ydata: (gaussian(xdata, *p) - ydata)
     
        optim1, success2 = optimize.leastsq(errfunc1, guess1[:], args=(data[:,0], data[:,1]))
        
        pl.plot(data[:,0], data[:,1], lw=5, c='g', label='measurement')
        pl.plot(data[:,0], gaussian(data[:,0], *optim1),
            lw=3, c='b', label='fit of 1 Gaussian')
        pl.legend(loc='best')
        pl.show()

        
        
    print optim1
    return optim1[1], optim1[2]

    
#     band_start = int(touch_pts - band_width * 2)
#     band_end = int(touch_pts + band_width * 2)
# 
#     
#     pl.plot(theta_bord[band_start:band_end], edge_int[band_start:band_end], '*')
#     pl.title('Intensity vs angle')
#     pl.xlabel('angle')
#     pl.ylabel('intensity')
#     pl.xlim(0,360)
#     pl.show()
#     
#     ########################################################
#     # Remove the anomalies
#     new_stdevs = np.delete(stdevs, range(band_start, band_end + 1))
#     print "number of deleted elements", len(stdevs) - len(new_stdevs)
#     print "mean spread from the edge", np.mean(new_stdevs)


def get_radius(image, theta, centre, rad_min, sigma):
    """
    Inputs:
        Image containing circle/edge
        (NB: Image must be in a lighter background for it to work!
        Otherwise change MAX to MIN if it is the opposite!)
        The angle through which the line will be drawn
        Centre of the circle
    
    Output:
        The value of the radius obtained from the image
        
    This function draws a line from the centre and check how that line changes.
    At every increment the pixel value along the line is stored.
    Then at the point where pixel values suddenly change we can assume that
    we reached the edge and at that point we will get the value of the radius.
    """
    
    Xc = centre[0]
    Yc = centre[1]
    
    # Denoise the image
    
    # Get values of pixels according to an increasing radius for a same angle
    # Get the radius to be at most half the image since the circle can't be bigger
    R = min(image.shape[0] / 2, image.shape[1] / 2) - 2
    
    # Simple trig identities
    # R is the max value that we can reach
    delta_x = R * math.sin(theta)
    delta_y = R * math.cos(theta)
    points = []

    # Go from 0 to 1.001 in steps of 0.001
    for alpha in np.arange(0, 1.001, 0.001):
        if np.sqrt((alpha*delta_x)**2 + (alpha*delta_y)**2) > rad_min:
            # Xc and Yc are the positions from the center
            # points stores all the points from the center going along the radius
            points.append(image[int(Xc + alpha * delta_x), int(Yc + alpha * delta_y)])

    rad_guess = image.shape[1] / 1.2 - rad_min
    
    optim1 = fit_gaussian_to_signal(points, sigma, True, rad_guess)
    optim1 = abs(optim1)
    
    data = np.array([range(len(points)), points]).T
    if theta >= 280 / 180. * 3.14 :
        pl.plot(data[:,0], data[:,1], lw=5, c='g', label='measurement')
        pl.plot(data[:,0], gaussian(data[:,0], *optim1),
            lw=3, c='b', label='fit of 1 Gaussian')
        pl.legend(loc='best')
        pl.show()

    if points[int(optim1[1])] >= (np.mean(points) + np.std(points) * 3):
        index_edge = optim1[1]
    else:
        index_edge = optim1[1]
    # Calculate the radius
    radius_sphere = index_edge * 0.001 * R + rad_min
    
    return round(radius_sphere, 4), points[int(index_edge)], optim1[2]


def close_to_Hough(rad, rad_h, thresh):
    """
    Check if the detected radius is close to
    the radius from the Hough transform
    """
    
    if np.allclose(rad, rad_h, 0, thresh):
        return rad
    else:
        return rad_h
    

def plot_radii(image, sigma = 3):
    """
    Inputs:
        Image as for the previous function
        Center as for the previous function
        
    Output:
        An array of radii at every angle position
        
    This function uses the previous function, but evaluates it
    through 360 angle. These radii are then plotted against angle
    and the variation can be seen.
    
    If it is constant then we can assume perfect reconstruction.
    """
    
    from skimage import color
    from skimage.util import img_as_ubyte
    from skimage import measure
    from skimage.measure import label
    from skimage.filter import threshold_otsu
    
#     image = np.pad(image, 10, 'edge')    
#     label_image = label(image)
    
#     for info in measure.regionprops(label_image,\
#                  ['major_axis_length', 'Centroid', 'BoundingBox', 'minor_axis_length']):
#         
#         centre = info['Centroid']
#         bound = info['BoundingBox']
#         major_axis = info['major_axis_length']
#         minor_axis = info['minor_axis_length']
#         
#     # if the radii are way off then it is an ellipse
#     # hence, simply return this value
#     if abs((bound[2] - bound[0]) / 2.0 - (bound[3] - bound[1]) / 2.0) > 3: 
#         return (False, major_axis / 2.0, minor_axis / 2.0)
         

#     new_image = img_as_ubyte(image)
#     new_image = color.gray2rgb(new_image)
#     new_image[centre[0], centre[1]] = (220, 20, 20)
#     
#     pl.imshow(new_image)
#     pl.show()
    filter = threshold_otsu(image)
    image = (image >= filter) * 1
    
    pl.imshow(image)
    pl.show()
    
    centre = (image.shape[0] / 2.0, image.shape[1] / 2.0)
    
    # Calculate radii for every angle
    radii_circle = []
    edge_int = []
    stdevs = []
    contact_points = []
    
    theta_bord = np.arange(0, 360, 1)
    
    rad_min = image.shape[1] / 3
    for theta in theta_bord:
        
        theta_pi = (math.pi * theta) / 180.0
        rad, intensity, width = get_radius(image, theta_pi, centre, rad_min, sigma)
        
        rad = close_to_Hough(rad, image.shape[1] / 1.2 / 2., 10)

        radii_circle.append(rad)
        edge_int.append(intensity)
        stdevs.append(width)

    
    # Plot radius
    pl.plot(theta_bord, radii_circle, '*')
    pl.xlabel('angle')
    pl.ylabel('radius')
    pl.xlim(0, 360)
    pl.show()
      
    pl.plot(theta_bord, edge_int, '*')
    pl.title('Intensity vs angle')
    pl.xlabel('angle')
    pl.ylabel('intensity')
    pl.xlim(0,360)
    pl.show()
    
    
    ########################################################
    # dont include points of contact for the qualiy assesment
#     pl.plot(new_theta_bord, new_edge_int, '*')
#     pl.title('Intensity vs angle')
#     pl.xlabel('angle')
#     pl.ylabel('intensity')
#     pl.xlim(0,360)
#     pl.show()

#     touch_pts = get_touch_point(edge_int, sigma)
    
#     if touch_pts != []:
#         new_edge_int = np.delete(edge_int, touch_pts)
#         new_theta_bord = np.delete(theta_bord, touch_pts)
#         new_stdevs = np.delete(stdevs, touch_pts)
#         print np.mean(stdevs)
#         print np.mean(new_stdevs)
#         return (True, np.mean(radii_circle), np.mean(new_stdevs))
#     else:
#         print "No touch points"
#         return (True, np.mean(radii_circle), np.mean(stdevs))
#     


import mhd_utils_3d as md

image_area1, meta_header = md.load_raw_data_with_mhd("/dls/tmp/jjl36382/complicated_data/spheres/sphere_hessian1/gradientgauss.mhd")
# image_area2, meta_header = md.load_raw_data_with_mhd("/dls/tmp/jjl36382/complicated_data/spheres/spheres_for_testing/sigma3.mhd")
# image_area3, meta_header = md.load_raw_data_with_mhd("/dls/tmp/jjl36382/complicated_data/spheres/spheres_for_testing/sigma6.mhd")

# pl.imshow(image_area1[190, :, :])
# pl.gray()
# pl.show()
mtf1 = plot_radii(image_area1[190, :, :], 1)

# pl.imshow(image_area2[190,:,:])
# pl.show()
mtf2 = plot_radii(image_area2[190, :, :], 3)
 
# pl.imshow(image_area3[190,:,:])
# pl.show()
mtf3 = plot_radii(image_area3[190, :, :], 6)
