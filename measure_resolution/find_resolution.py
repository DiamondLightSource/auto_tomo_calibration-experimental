import numpy as np
import pylab as pl
from scipy.ndimage.filters import median_filter
from skimage.filter import threshold_otsu
from skimage import io, exposure
import os
import fit_data
from scipy import misc
from math import ceil


def save_data(filename, data):
    import cPickle
    print("Saving data")
    f = open(filename, 'w')
    cPickle.dump(data, f)
    f.close()


def create_dir(directory):
    
    if not os.path.exists(directory):
        os.makedirs(directory)


def fit_and_visualize(image, folder_name, r1, r2, window_size):
    """
    Takes in the region of interest, which is a 2D image.
    
    Modulation is calculated for the lines further away
    from the touch point. It is used for normalizing MTF.
    
    Intensity_left/right store the intensities of the left and right
    spheres, which are the mean pixel values over a narrow strip
    along every sphere.
    """
    # Denoise using a median filter
    if window_size != 0:
        denoised = median_filter(image, window_size)
    else:
        denoised = image
  
    # Save the images containing the gaps
    misc.imsave(folder_name + "touch_img.png", denoised)
    misc.imsave(folder_name + "touch_img.tif", denoised)
    
    # Calculate average sphere intensity. They are segmented using
    # thresholding and the pixels are averaged
    left = denoised[:, 0:image.shape[1] / 2. - 10]
    right = denoised[:, image.shape[1] / 2. + 10:image.shape[1]]
    
    thresh_l = threshold_otsu(left)
    thresh_r = threshold_otsu(right)
    
    sphere_pixels_l = []
    for y in range(left.shape[1]):
        for x in range(left.shape[0]):
            pixel = left[x, y]
            if pixel > thresh_l:
                sphere_pixels_l.append(pixel)
    
    sphere_pixels_r = []
    for y in range(right.shape[1]):
        for x in range(right.shape[0]):
            pixel = right[x, y]
            if pixel > thresh_r:
                sphere_pixels_r.append(pixel)
                
    intensity_left = np.mean(sphere_pixels_l)
    intensity_right = np.mean(sphere_pixels_r)
    
    # This is the modulation at a high separation, used
    # to normalize the MTF values between 0% and 100%
    low_freq_left = (intensity_left - np.min(denoised)) /\
                    (intensity_left + np.min(denoised))
    low_freq_right = (intensity_right - np.min(denoised)) /\
                     (intensity_right + np.min(denoised))
    
    gap = []
    mtf_cleft = []
    mtf_cright = []
    
    # Take values only to the middle of the image
    # since the problem is symmetric on the other side
    for i in np.arange(0., image.shape[0] / 2.):
         
        Xdata = []
        Ydata = []
        gapX = []
        gapY = []
        
        distance = dist_between_spheres(r1, r2, i, image.shape[0] / 2.)
        
        signal = [pixel for pixel in denoised[i, :]]
        gap_signal = []
        
        
        for j in np.arange(0., image.shape[1]):
            # Used to plot line on the image
            Xdata.append(j)
            Ydata.append(i)
            
            # If we are in the region where the spheres are separated,
            # stores these vlaues to plot the gap
            if image.shape[1] / 2. + distance / 2. > j > image.shape[1] / 2. - distance / 2.:
                gapX.append(j)
                gapY.append(i)
            
            # Take the region around the gap, which later on will be used
            # to define the intensity at the gap between the spheres.
            # The width of the gap is not exact
            if image.shape[1] / 2. + distance + 10 > j > image.shape[1] / 2. - distance - 10:
                gap_signal.append(denoised[i, j])

        # Check if the gap still exists
        if gap_signal:
            # If the signal minima is higher than the minima in the gap
            # it means that the contrast must be lost in the centre - stop
            if distance < 10:
                if np.min(signal) >= np.min(gap_signal):
                    
                    mtf = 100 * modulation(np.min(gap_signal), intensity_left, distance) / low_freq_left
    
                    # PLOT THE REGION AROUND THE MIDDLE OF THE CURVE
                    # PLOT THE LINE PROFILE
                    # Do this only if mtf is mroe than 9, bellow that,
                    # the gap is unresolved, and gaussian width starts
                    # to spread out - ruins the results
                    if mtf >= 1:
                        # FIT A GAUSSIAN
                        amp = -np.min(gap_signal)
                        centre = np.mean(np.argwhere(np.min(gap_signal) == gap_signal))
                        sigma = distance / 6.
                        offset = np.max(gap_signal)
                        
                        guess_params = [amp, centre, sigma, offset]
                        Xfit, Yfit, fwhm, fit_centre = fit_data.GaussConst(gap_signal, guess_params)
                        
                        ymax = np.max(denoised)
                        ymin = np.min(denoised)
                        data = np.array([range(len(gap_signal)), gap_signal]).T
                        pl.plot(data[:,0],
                                data[:,1], 'bo')
                        pl.plot(Xfit, Yfit)
                        pl.title("Analytical {0} / Fitted dist {1} / Contrast {2} ".format(round(distance, 2), round(fwhm, 2), round(mtf,2)))
                        pl.ylim(ymin, ymax)
                        
                        # PLOT THE ANALYTICAL WIDTH
                        pl.plot(np.repeat(fit_centre - distance / 2., len(Yfit)),
                                np.arange(len(Yfit)), 'r-')
                        pl.plot(np.repeat(fit_centre + distance / 2., len(Yfit)),
                                np.arange(len(Yfit)), 'r-', label = "Analytical")
                        
                        pl.legend()
                        pl.savefig(folder_name + 'results%i.png' % i)
                        pl.close('all')
                       
                        # Store the values of the gap width for every value
                        # of contrast
                        gap.append(distance)
                        
                        mtf = 100 * modulation(np.min(gap_signal), intensity_left, distance) / low_freq_left
                        mtf_cleft.append(mtf)
                    
                        mtf = 100 * modulation(np.min(gap_signal), intensity_right, distance) / low_freq_right
                        mtf_cright.append(mtf)


    ############# LEFT SPHERE #########################
    pl.gca().invert_xaxis()
    pl.plot(gap, mtf_cleft, 'r', label="left sphere data")
    pl.xlabel("Width in pixels")
    pl.ylabel("MTF %")
    pl.tight_layout()
    
    # Save data points
    save_data(folder_name + 'gap_width.npy', gap)
    save_data(folder_name + 'mtf_cleft.npy', mtf_cleft)
    
    f = open(folder_name + 'gap_width.txt', 'w')
    for i in range(len(gap)):
        f.write(repr(gap[i]) + '\n')
    f.close()
    
    f = open(folder_name + 'mtf_cleft.txt', 'w')
    for i in range(len(mtf_cleft)):
        f.write(repr(mtf_cleft[i]) + '\n')
    f.close()
    
    pl.savefig(folder_name + 'mtf_left.png')
    pl.close('all')
    
    
    ############### RIGHT SPHERE #####################

    pl.gca().invert_xaxis()
    pl.plot(gap, mtf_cright, 'r', label="left sphere data")
    pl.xlabel("Width in pixels")
    pl.ylabel("MTF %")
    
    # Save data points
    save_data(folder_name + 'mtf_cright.npy', mtf_cright)
    
    f = open(folder_name + 'mtf_cright.txt', 'w')
    for i in range(len(mtf_cright)):
        f.write(repr(mtf_cright[i]) + '\n')
    f.close()
    
    pl.savefig(folder_name + 'mtf_right.png')
    pl.close('all')


def dist_between_spheres(r1, r2, Y, C):
    """
    Calculate distance between the spheres using
    geometry. Read report to see how it is done.
    """
    h = C - Y
    
    d1 = np.sqrt(r1**2 - h**2)
    d2 = np.sqrt(r2**2 - h**2)

    dist = r1 - d1 + r2 - d2
    
    return dist


def modulation(minima, contrast, distance):
    """
    modulation(contrast) = (Imax - Imin) / (Imax + Imin)
    """
    
    numerator = contrast - minima
    denominator = contrast + minima
    
    return numerator / denominator


def modulus(vect):
    """
    Get the modulus of a vector
    """
    return np.sqrt(vect[0]**2 + vect[1]**2 + vect[2]**2)


def distance_3D(c1, c2):
    """
    Calculate the distance between two points
    """
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)


def vector_3D(pt1, pt2, t):
    """
    Compute the 3d line equation in a parametric form
    
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
    between pt1 and pt2 AND such that lies in x-y plane
    at height Z
    
    'which' describes through which point to draw it (pt1 or pt2)
    
    Sx describes the position along the perpendicular vector.
    """

    v = ((pt2[0] - pt1[0]), (pt2[1] - pt1[1]), (pt2[2] - pt1[2]))
    
    if which == 1:
        Sx, Sy = (pt1[0] - v[1] / np.sqrt(v[0]**2 + v[1]**2) * Sx,
                  pt1[1] + v[0] / np.sqrt(v[0]**2 + v[1]**2) * Sx)
        Sz = pt1[2]
        
    elif which == 2:
        Sx, Sy = (pt2[0] - v[1] / np.sqrt(v[0]**2 + v[1]**2) * Sx,
                  pt2[1] + v[0] / np.sqrt(v[0]**2 + v[1]**2) * Sx)
        Sz = pt2[2]
        
    return [Sx, Sy, Sz + Z]



def vector_perpendicular_ct_pt(pt1, pt2, r1, Sx):
    """
    Vector must be perpendicular to the one
    connecting the centres of the spheres, v1, and
    to the vector perpendicular to v1, that goes
    throught he point of contact
    """

    v = ((pt2[0] - pt1[0]), (pt2[1] - pt1[1]), (pt2[2] - pt1[2]))
    ct_pt = vector_3D(pt1, pt2, r1)
    perp_v_in_xy = np.array(vector_perpendicular_3D(ct_pt, pt2, 1, 0, -1)) -\
                    np.array(vector_perpendicular_3D(ct_pt, pt2, 1, 0, 1))
    
    vect = np.cross(v, perp_v_in_xy)
    mod_vect = modulus(vect)
    
    x = ct_pt[0] + vect[0] / mod_vect * Sx
    y = ct_pt[1] + vect[1] / mod_vect * Sx
    z = ct_pt[2] + vect[2] / mod_vect * Sx
    
    return [x, y, z]
    # Find perpendicular vector components
#     if np.isinf(1. / np.sqrt(v[0]**2 + v[2]**2)):
#         v1 = np.array([ct_pt[0],
#                         ct_pt[1] - v[2] / np.sqrt(v[1]**2 + v[2]**2) * Sx,
#                         ct_pt[2] + v[1] / np.sqrt(v[1]**2 + v[2]**2) * Sx])
#     
#     elif np.isinf(1. / np.sqrt(v[1]**2 + v[2]**2)):
#         v1 = np.array([ct_pt[0] - v[2] / np.sqrt(v[0]**2 + v[2]**2) * Sx,
#                         ct_pt[1],
#                         ct_pt[2] + v[0] / np.sqrt(v[0]**2 + v[2]**2) * Sx])
#     else:
#         v1 = np.array([ct_pt[0] - v[2] / np.sqrt(v[0]**2 + v[2]**2) * Sx,
#                         ct_pt[1] - v[2] / np.sqrt(v[1]**2 + v[2]**2) * Sx,
#                         ct_pt[2] + v[0] / np.sqrt(v[0]**2 + v[2]**2) * Sx])
#     
#     # Add them to get the final vector
#     vector_sum = v1 + v2
#
#     return v1


# v1 = (0, 0, 0)
# v2 = (5, 0, 5)
# 
# vector1 = [vector_3D(v1, v2, i) for i in range(5)]
# 
# vector2 = [vector_perpendicular_ct_pt(v1, v2, 1, i) for i in np.arange(5)]
# 
# print vector1
# print vector2

    
def project_onto_plane(vect):
    """
    Return vector projection onto the xy plane
    """
    x, y, z = vect
    
    return (x, y, 0.)


def find_contact_3D(centroids, radius, tol = 20.):
    """
    Arrays of all the centroids and all radii
    tol defines the error tolerance between radii distance
    
    Check all centre pairs and determine,
    based on their radii, if they are in contact or not
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
            print ""
            print "Difference between radii sum and centre distance is", abs(D - L)
            print "Distance is ", L
            print "Radii sum is ", D
            print ""
            if abs(D - L) <= tol:
                
                touch_pt = vector_3D(c1, c2, r1)
                touch_pts.append(touch_pt)
                centres.append((c1, c2))
                radii.append((r1, r2))
                
    return centres, touch_pts, radii



def sample_rate(P2, P1):
    """
    When we have to loop through pixels at
    an angle, the angle needs to be taken
    into account. This calculates the change in distance
    depending on where the vector is pointing
    """
    from numpy import (array, dot)
    from numpy.linalg import norm
    
    v = np.array([P1[0] - P2[0], P1[1] - P2[1], P1[2] - P2[2]])
    normal = np.array([0,0,1])
    
    projection = np.cross(normal, np.cross(v,normal))
    
    c = np.dot(v, projection) / modulus(projection) / modulus(v)
    
    return 1. / c


def get_slice(P1, P2, name):
    """
    Get slice through centre for analysis
    """

    centre_dist = distance_3D(P1, P2)
    sampling = sample_rate(P1, P2) - 1
    plot_img = np.zeros((centre_dist / 2. + 1, centre_dist + 1))
    
    Xrange = np.linspace(-centre_dist / 4., centre_dist /4.,
                            centre_dist / 2. + 1)
    Trange = np.linspace(0., centre_dist,
                            centre_dist * 2.)

    for time in Trange:
        # Go up along the line
        pt = vector_3D(P1, P2, time + sampling)
        interpolated = trilinear(name, pt)

        for X in Xrange:
    
            # Get along the X direction for every height
            x, y, z = vector_perpendicular_3D(pt, P2, 1, 0, X)

#             pixel_value = interpolation(x, y, img)
            pixel_value = interpolated([x, y, z])
            
            plot_img[X + centre_dist / 4., time] = pixel_value
                
    return plot_img


def get_slice_perpendicular(P1, P2, r1, name):
    """
    Finds a vector between the centres.
    Takes the point on the vector that is on the contact point.
    Finds a vector going through the contact point that is also
    perpendicular to the line connecting the centres.
    
    The slice is then reconstructed from several images
    """
    
    # time goes along the vector between P1 and P2
    # since it might be at an angle, I can't loop in 1
    # pixel increments - this will miss certain slices. Therefore,
    # I need to loop through by 1/cosA, where A is angle between
    # the xy plane and vector P1->P2
    centre_dist = distance_3D(P1, P2)

    perp1 = vector_perpendicular_ct_pt(P1, P2, r1, centre_dist /4.)
    perp2 = vector_perpendicular_ct_pt(P1, P2, r1, -centre_dist /4.)
    sampling = sample_rate(perp1, perp2) - 1
    
    plot_img = np.zeros((np.int(np.round(centre_dist / 2. + 1, 0)), np.int(np.round(centre_dist / 2. + 1, 0))))
    
    Xrange = np.linspace(-centre_dist / 4., centre_dist /4.,
                            centre_dist / 2. + 1)
    Trange = np.linspace(-centre_dist / 4., centre_dist /4.,
                            centre_dist / 2. + 1)

    for time in Trange:
        # Go up along the line
        pt = vector_perpendicular_ct_pt(P1, P2, r1, time + sampling)
        interpolated = trilinear(name, pt)
        
        for X in Xrange:
    
            # Get along the X direction for every height
            x, y, z = vector_perpendicular_3D(pt, P2, 1, 0, X)

#             pixel_value = interpolation(x, y, img)
            pixel_value = interpolated([x, y, z])
            
            plot_img[X + centre_dist / 4., time + centre_dist / 4.] = pixel_value
    
                   
    return plot_img


def check_alignment(image, r1, r2):
    """
    Take a particular line though the image and check
    if the spheres were properly aligned in the z direction.
    It happens to be off by a pixel or two sometimes
    """
                
    distance = dist_between_spheres(r1, r2, image.shape[0] / 2. + 10, image.shape[0] / 2.)
    gap_signal = []
    denoised = median_filter(image.copy(), 3)
    
    for j in np.arange(0., image.shape[1]):
        # Take the region around the gap, which later on will be used
        # to define the intensity at the gap between the spheres.
        # The width of the gap is not exact
        if image.shape[1] / 2. + distance + 5 > j > image.shape[1] / 2. - distance - 5:
            gap_signal.append(denoised[image.shape[0] / 2. + 10, j])
    
    centre = np.mean(np.argwhere(np.min(gap_signal) == gap_signal))
    print centre
    print len(gap_signal) / 2.
    print
    
    if abs(centre - len(gap_signal) / 2.) <= 1.5:
        return True
    else:
        return False


def interpolation(x, y, img):
    """
    http://en.wikipedia.org/wiki/Bilinear_interpolation
    """
    from math import floor, ceil
    
    x1 = ceil(x)
    x2 = floor(x)
    y1 = ceil(y)
    y2 = floor(y)
    
    Q11 = (x1, y1)
    Q12 = (x1, y2)
    Q21 = (x2, y1)
    Q22 = (x2, y2)
    
    f11 = img[Q11[0], Q11[1]]
    f12 = img[Q12[0], Q12[1]]
    f21 = img[Q21[0], Q21[1]]
    f22 = img[Q22[0], Q22[1]]

    try:
        pixel_value = 1 / ((x2 - x1) * (y2 - y1)) * (f11 * (x2 - x) * (y2 - y) +
                                                 f21 * (x - x1) * (y2 - y) +
                                                 f12 * (x2 - x) * (y - y1) +
                                                 f22 * (x - x1) * (y - y1))
    except:
        pixel_value = np.mean([f11, f12, f21, f22])
        
    return pixel_value


def trilinear(name, pt):
    """
    Trilinear interpolation
    http://docs.scipy.org/doc/scipy-dev/reference/generated/
    scipy.interpolate.RegularGridInterpolator.html
    """
    from scipy.interpolate import RegularGridInterpolator

    input_file = name % int(np.floor(pt[2]))
    data0 = io.imread(input_file)
    
    input_file = name % int(np.ceil(pt[2]))
    data1 = io.imread(input_file)
    
    xdim = data0.shape[0]
    ydim = data0.shape[1]
    zdim = 2
    
    empty_arr = np.empty((xdim, ydim, zdim))
    
    empty_arr[:, :, 0] = data0
    empty_arr[:, :, 1] = data1
    
    x = np.linspace(0, xdim - 1, xdim)
    y = np.linspace(0, ydim - 1, ydim)
    z = np.linspace(int(np.floor(pt[2])), int(np.ceil(pt[2])), zdim)
    
    
    interp = RegularGridInterpolator((x, y, z), empty_arr)
    
    return interp


# def load_up_images(pt1, pt2, name):
#     """
#     Stack up the region of interest from image slices
#     """
#     zstart = int(np.min(pt1[2], pt2[2]))
#     zend = int(np.max(pt1[2], pt2[2]))
#     
#     xdim = 
#     ydim
#     
#     input_file = name % zstart
#     data = io.imread(input_file)
#     
#     zrange = np.linspace(zstart, zend, zend-zstart)
#     store_ROI = np.empty((data.shape[0]))
#     
#     for i in zrange:
#         input_file = name % i
#         data = io.imread(input_file)
#     
#     return


def touch_lines_3D(pt1, pt2, folder_name, name, r1, r2, window_size):
    """
    Goes along lines in the region between
    two points.
    Used for obtaining the widths of the gaussian fitted
    to the gap between spheres
    """

    # Create an array to store the slanted image slices
    # used for plotting
    L = distance_3D(pt1, pt2)
    D = r1 + r2
    print ""
    print "Difference between radii sum and centre distance is", abs(D - L)
    print "Distance is ", L
    print "Radii sum is ", D
    print ""
    create_dir(folder_name + "plots/")
    
    perpendicular_slice = get_slice_perpendicular(pt1, pt2, r1, name)
    misc.imsave(folder_name + "perp_slice.tif", perpendicular_slice)
    print "saving the perpendicular slice"
    
    ROI = get_slice(pt1, pt2, name)
    fit_and_visualize(ROI, folder_name + "plots/", r1, r2, window_size)
    print "saving the slice for MTF"

    return

# import cPickle
# import pylab as pl
#               
# f = open("/dls/tmp/jjl36382/50867/plots/(793.0, 1143.07, 801.86),(682.61, 1141.0, 1410.12)/plots/gap_width.npy", 'r')
# x1 = cPickle.load(f)
# f.close()
# f = open("/dls/tmp/jjl36382/50873/plots/(796.04, 1146.95, 806.3),(685.0, 1143.98, 1414.78)/plots/gap_width.npy", 'r')
# x2 = cPickle.load(f)
# f.close()
# f = open("/dls/tmp/jjl36382/50880/plots/(798.04, 1147.99, 811.83),(685.0, 1143.0, 1418.03)/plots/gap_width.npy", 'r')
# x3 = cPickle.load(f)
# f.close()
# f = open("/dls/tmp/jjl36382/50867/plots/(793.0, 1143.07, 801.86),(682.61, 1141.0, 1410.12)/plots/mtf_cleft.npy", 'r')
# y1 = cPickle.load(f)
# f.close()
# f = open("/dls/tmp/jjl36382/50873/plots/(796.04, 1146.95, 806.3),(685.0, 1143.98, 1414.78)/plots/mtf_cleft.npy", 'r')
# y2 = cPickle.load(f)
# f.close()
# f = open("/dls/tmp/jjl36382/50880/plots/(798.04, 1147.99, 811.83),(685.0, 1143.0, 1418.03)/plots/mtf_cleft.npy", 'r')
# y3 = cPickle.load(f)
# f.close()
#                    
# pl.plot(x1, y1, 'r', label = "53keV")
# pl.plot(x3, y3, 'g', label = "75keV")
# pl.plot(x2, y2, 'b', label = "130keV")
#             
#               
# pl.xlabel("Distance between spheres (pixels)")
# pl.ylabel("MTF %")
# pl.legend()
# pl.gca().invert_xaxis()
# pl.savefig("./median_0.tif")
# pl.show()
# #  


