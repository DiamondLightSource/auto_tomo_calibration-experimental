import pylab as pl
import numpy as np

import math
from math import pi, log

from scipy import optimize
from scipy.signal import argrelextrema
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
    Zc = int(image.shape[2] / 2.)
    R = image.shape[0] / 4.
    
    
    delta_x = R * np.sin(0) * np.cos(0)
    delta_y = R * np.sin(0) * np.sin(0)
    delta_z = R * np.cos(0)
    points = []
    # Go from 0 to 1.001 in steps of 0.001
    for alpha in np.arange(0, 1.001, 0.001):
            # Xc and Yc are the positions from the center
            # points stores all the points from the center going along the radius
            points.append( image[int(Xc + alpha * delta_x), int(Yc + alpha * delta_y), int(Zc + alpha * delta_z)] )
            
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


def fit_gaussian_to_signal(points, sigma, rad_pos = 0., height = 0., width_guess = 10.):
    """
    Detects peaks
    First guess for the Gaussian is the firs maxima in the signa;
    """
    rms_noise5 = rms_noise_signal(points) * 3
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


def get_touch_point(intensity, sigma, noise):
    
    
    peaks_max, peaks_min = peakdetect(intensity,
                       lookahead = 180 / 4, delta = noise * 6)
    if peaks_min != []:

        outliers = is_outlier(intensity, thresh=3)
        return outliers
    else:
        return []
#         pl.plot(data[:,0], data[:,1], lw=5, c='g', label='measurement')
#         pl.plot(data[:,0], gaussian(data[:,0], *optim1),
#             lw=3, c='b', label='fit of 1 Gaussian')
#         pl.legend(loc='best')
#         pl.show()

        
        
    
    

def get_radius(image, theta, phi, centre, rad_min, sigma):

    pl.close('all')
    
    # Plot value of pixel as a function of radius
    Xc = centre[0]
    Yc = centre[1]
    Zc = centre[2]
    
    R = min(image.shape[0] / 2, image.shape[1] / 2, image.shape[2] / 2) - 1
    
    delta_x = R * np.sin(phi) * np.cos(theta)
    delta_y = R * np.sin(phi) * np.sin(theta)
    delta_z = R * np.cos(phi)
    
    points = []
    
    step = 0.001
    
    for alpha in np.arange(0,1 + step, step):
        if np.sqrt((alpha*delta_x)**2 + (alpha*delta_y)**2 + (alpha*delta_z)**2) > rad_min:
            points.append( image[int(Xc + alpha * delta_x), int(Yc + alpha * delta_y), int(Zc + alpha * delta_z)] )
    
    rad_guess = image.shape[1] / 1.2 - rad_min
    
    
    optim1 = fit_gaussian_to_signal(points, sigma, rad_guess)
    optim1 = abs(optim1)
    
#     data = np.array([range(len(points)), points]).T
#     if theta >= 150 / 180. * 3.14 :
#         pl.plot(data[:,0], data[:,1], lw=5, c='g', label='measurement')
#         pl.plot(data[:,0], gaussian(data[:,0], *optim1),
#             lw=3, c='b', label='fit of 1 Gaussian')
#         pl.legend(loc='best')
#         pl.show()

#     if points[int(optim1[1])] >= (np.mean(points) + np.std(points) * 3):
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
    

def plot_radii(image_area, centre, start, stop, step, sigma = 1):
    
    # Calculate radii for every angle 
    # initial guess for the radius to make calculations faster
    from skimage.filter import threshold_otsu
    
    filter = threshold_otsu(image_area)
    image = (image_area >= filter) * 1
    
    rad_min = image_area.shape[2] / 3
    rad_Hough = image_area.shape[1] / 2. / 1.2
    
    theta_bord = np.arange(start, stop, step)
    phi_bord = np.arange(0, 180, step)
    
    radii_sphere = np.zeros((len(theta_bord), len(phi_bord)))
    new_radii_sphere = np.zeros((len(theta_bord), len(phi_bord)))
    lsf_height = np.zeros((len(theta_bord), len(phi_bord)))
    new_lsf_width = np.zeros((len(theta_bord), len(phi_bord)))
    contact_pts = np.zeros((len(theta_bord), len(phi_bord)))
    lsf_width = np.zeros((len(theta_bord), len(phi_bord)))
    
    for theta in theta_bord:
        angle1 = (theta - start) / step
        
        for phi in phi_bord:
            
            theta_rad = np.radians(theta)
            phi_rad = np.radians(phi)
            angle2 = phi / step
            
            # get data from line tracing
            rad, height, width = get_radius(image_area, theta_rad, phi_rad, centre, rad_min, sigma)
            
           
            if height != 0:
                radii_sphere[angle1, angle2] = rad
                lsf_height[angle1, angle2] = height
                lsf_width[angle1, angle2] = width
            else:
                radii_sphere[angle1, angle2] = rad_Hough
                lsf_height[angle1, angle2] = height
                lsf_width[angle1, angle2] = width
                contact_pts[angle1, angle2] = 1
                
#     pl.plot(phi_bord, radii_sphere.T, '*')
#     pl.xlabel('angle')
#     pl.ylabel('radius')
#     pl.show()
#     
    return radii_sphere, contact_pts, lsf_width
 
 
 
# import mhd_utils_3d as md
#         
# image_area, meta_header = md.load_raw_data_with_mhd("/dls/tmp/jjl36382/complicated_data/spheres/sphere_hessian1/gradientgauss.mhd")
# # pl.imshow(image_area)
# # pl.show()
#         
#         
# print image_area.shape[0]
# print image_area.shape[1]
# print image_area.shape[2]
#               
# centre = (int(128 * 1.2), int(128 * 1.2), int(128 * 1.2))
# start = 250.
# stop = 251.
# step = 1
#              
# plot_radii(image_area, centre, start, stop, step, 1)