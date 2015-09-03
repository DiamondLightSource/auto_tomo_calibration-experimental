import numpy as np
from skimage.draw import circle_perimeter
from skimage.transform import hough_circle
import pylab as pl
from scipy import ndimage, misc
import mhd_utils_3d as md

from skimage import io
from skimage import measure
from skimage import exposure
from scipy.ndimage.filters import median_filter
from skimage.morphology import watershed
from skimage.filter import threshold_otsu, sobel, denoise_tv_chambolle
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
from sphere_fit import leastsq_sphere
from circle_fit import leastsq_circle

def save_data(filename, data):
    print("Saving data")
    f = open(filename, 'w')
    np.save(f, data)
    f.close()
    

def get_edges(np_image):
    # Get the coordinates of the circle edges
    X = []
    Y = []
    Z = []
    
    circles = []  # to get the outlines of the circles
    C = []  # to get the centres of the circles, in relation to the different areas
    R = []  # to get radii
    
    coords = np.column_stack(np.nonzero(np_image))
    X = np.array(coords[:, 0])
    Y = np.array(coords[:, 1])

    # Fit a circle and compare measured circle area with
    # area from the amount of pixels to remove trash
    XC, YC, RAD, RESID = leastsq_circle(X, Y)

    # Hough radius estimate
    hough_radii = np.arange(RAD - RAD / 2., RAD + RAD / 2.)
    
    # Apply Hough transform
    hough_res = hough_circle(np_image, hough_radii)
         
    centers = []
    accums = []
    radii = []
    
    img = np.zeros_like(np_image)
    
    # For each radius, extract one circle
    for radius, h in zip(hough_radii, hough_res):
        peaks = peak_local_max(h, num_peaks=2)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius, radius])
         
    for idx in np.argsort(accums)[::-1][:1]:
        center_x, center_y = centers[idx]
        C.append((center_x, center_y))
        radius = radii[idx]
        R.append(radius)
        cx, cy = circle_perimeter(int(round(center_x, 0)),
                                  int(round(center_y, 0)),
                                  int(round(radius, 0)))
        circles.append((cx, cy))
        
    if C:
        for cent in C:
            xc, yc = cent
            np_image[int(xc), int(yc)] = 255
            np_image[int(xc)-5:int(xc)+5, int(yc)-5:int(yc)+5] = 255
    
    if circles:
        for per in circles:
            e1, e2 = per
            np_image[e1, e2] = 255
    
    return [C, R, circles, []], np_image
# 
# imp = io.imread("/dls/tmp/jjl36382/50873/spheres/sphere_tif1/slice450.tif")
# 
# get_edges(imp)
