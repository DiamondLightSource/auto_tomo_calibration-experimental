from scipy.ndimage import filters
import numpy as np
import pylab as pl

def compute_harris_response(im, sigma=3):
    """ Compute the Harris corner detector response function
    for each pixel in a graylevel image.
    
    This indicator function allows to distinguish different
    eigenvalue relative sizes without computing them
    directly. It is also called the Harris Stephens function
     """
    
    # derivative in x direction of the image
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    
    # derivative in y direction of the image
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
    
    # compute components of the Harris matrix
    # it is usually weighted by a Gaussian
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)
    
    # determinant and trace
    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy
    
    return Wdet / Wtr


def hessian_response(im, sigma=3):
    
    # derivative in x direction of the image
    imx = np.zeros(im.shape)
    imxx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    filters.gaussian_filter(imx, (sigma, sigma), (0, 1), imxx)
    
    # derivative in y direction of the image
    imy = np.zeros(im.shape)
    imyy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
    filters.gaussian_filter(imy, (sigma, sigma), (1, 0), imyy)
    
    imxy = np.zeros(im.shape)
    filters.gaussian_filter(imx, (sigma, sigma), (1, 0), imxy)
    
    Wxx = filters.gaussian_filter(imxx, sigma)
    Wxy = filters.gaussian_filter(imxy, sigma)
    Wyy = filters.gaussian_filter(imyy, sigma)
    
    trace_hes = Wxx + Wyy
    det_hes = Wxx * Wyy - Wxy ** 2
    
    return det_hes / trace_hes

def get_harris_points(harrisim, min_dist=1, threshold=0.0005):
    """ Return corners from a Harris response image
    min_dist is the minimum number of pixels separating
    corners and image boundary. 
    
    min_dist specifies the allowed length"""
    
    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    
    # get coordinates of candidates
    coords = np.array(harrisim_t.nonzero()).T
    
    # ...and their values
    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    # sort candidates
    index = np.argsort(candidate_values)
    
    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
    
    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
            (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0
            
    return filtered_coords


def plot_harris_points(image, filtered_coords):
    """ Plots corners found in image. """
    pl.figure()
    pl.gray()
    pl.imshow(image)
    pl.show()
    x = [p[1] for p in filtered_coords]
    y = [p[0] for p in filtered_coords]
    
    for i in x:
        
    #pl.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords])
    #pl.axis("off")
    #pl.show()
