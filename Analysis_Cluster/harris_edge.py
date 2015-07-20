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


def hessian_response(im, sigma=3, threshold = 0.05):
    
#     # derivative in x direction of the image
#     imx = np.zeros(im.shape)
#     imxx = np.zeros(im.shape)
#     filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
#     filters.gaussian_filter(imx, (sigma, sigma), (0, 1), imxx)
#     
#     # derivative in y direction of the image
#     imy = np.zeros(im.shape)
#     imyy = np.zeros(im.shape)
#     filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
#     filters.gaussian_filter(imy, (sigma, sigma), (1, 0), imyy)
#     
#     imxy = np.zeros(im.shape)
#     filters.gaussian_filter(imx, (sigma, sigma), (1, 0), imxy)
#     
#     Wxx = filters.gaussian_filter(imxx, sigma)
#     Wxy = filters.gaussian_filter(imxy, sigma)
#     Wyy = filters.gaussian_filter(imyy, sigma)
# 
#     H = np.array([[Wxx, Wxy],
#                   [Wxy, Wyy]])
#     
    from numpy import linalg as LA
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    
    Hxx, Hxy, Hyy = hessian_matrix(im, sigma=0.1)
    e_big, e_small = hessian_matrix_eigvals(Hxx, Hxy, Hyy)
    
    #eiglast = 0.5 * (Wxx + Wyy + np.sqrt(Wxx**2 + 4*Wxy**2 - 2*Wxx*Wyy + Wyy**2 ))

#     det_hes = Wxx * Wyy - Wxy ** 2

    
    eiglast = e_big
    
    # get maxima of the determinant
#     det_thresh = eiglast.max() * threshold
#     det_bin = (eiglast >= det_thresh) * 1
#         
#     coordinates = np.array(det_bin.nonzero()).T
    
    x = [p[0] for p in coordinates]
    y = [p[1] for p in coordinates]
    
    edges = np.zeros(im.shape)
    edges[x,y] = 1
    pl.imshow(edges)
    pl.gray()
    pl.show()
    
    return 

recon = np.load("/dls/tmp/jjl36382/complicated_data/spheres/sphere1.npy")[:,:,100]
hessian_response(recon, 1, 0.5)

def hessian_response_3d(im, sigma=3, threshold = 0.005):
    
    import numpy as np
    
    # derivative in x direction of the image
    imx = np.zeros(im.shape)
    imxx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma, sigma), (0, 0, 1), imx)
    filters.gaussian_filter(imx, (sigma, sigma, sigma), (0, 1, 1), imxx)
    
    # derivative in y direction of the image
    imy = np.zeros(im.shape)
    imyy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma, sigma), (1, 0, 1), imy)
    filters.gaussian_filter(imy, (sigma, sigma, sigma), (1, 0, 1), imyy)
    
    imz = np.zeros(im.shape)
    imzz = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma, sigma), (1, 1, 0), imz)
    filters.gaussian_filter(imz, (sigma, sigma, sigma), (1, 1, 0), imzz)
    
    imxy = np.zeros(im.shape)
    filters.gaussian_filter(imx, (sigma, sigma, sigma), (1, 0, 1), imxy)
    
    imxz = np.zeros(im.shape)
    filters.gaussian_filter(imx, (sigma, sigma, sigma), (1, 1, 0), imxz)
    
    imyz = np.zeros(im.shape)
    filters.gaussian_filter(imy, (sigma, sigma, sigma), (1, 1, 0), imyz)
    
    
    Wxx = filters.gaussian_filter(imxx, sigma)
    Wxy = filters.gaussian_filter(imxy, sigma)
    Wxz = filters.gaussian_filter(imxz, sigma)
    Wzy = filters.gaussian_filter(imyz, sigma)
    Wyy = filters.gaussian_filter(imyy, sigma)
    Wzz = filters.gaussian_filter(imzz, sigma)

    H = np.array( [[Wxx, Wxy, Wxz],
                   [Wxy, Wyy, Wzy],
                   [Wxz, Wzy, Wzz]] )
    
    import numpy.linalg as LA
    
    e_val, e_vec = LA.eig(H)
    
    print H.shape
    
    
    det_hes = Wxx * (Wyy * Wzz - Wzy ** 2) - Wxy * (Wxy * Wzz - Wzy * Wxz) + Wxz * (Wxy * Wzy - Wyy * Wxz)
    
    # get maxima of the determinant
    det_thresh = det_hes.max() * threshold
    det_bin = (det_hes >= det_thresh) * 1
        
    coordinates = np.array(det_bin.nonzero()).T
    
    x = [p[0] for p in coordinates]
    y = [p[1] for p in coordinates]
    z = [p[2] for p in coordinates]
    
    edges = np.zeros(im.shape)
    edges[x,y, z] = 1
    pl.imshow(edges[:, :, 50])
    #pl.scatter(x, y)
    pl.gray()
    pl.show()
    
    return edges


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
    
    #for i in x:
        
    #pl.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords])
    #pl.axis("off")
    #pl.show()
