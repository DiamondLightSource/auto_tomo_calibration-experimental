import numpy as np
import pylab as pl
from scipy.ndimage.filters import median_filter
from skimage import io
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


def fit_and_visualize(image, folder_name, r1, r2):
    """
    Takes in the region of interest, which is a 2D image.
    
    Modulation is calculated for the lines further away
    from the touch point. It is used for normalizing MTF.
    
    Intensity_left/right store the intensities of the left and right
    spheres, which are the mean pixel values over a narrow strip
    along every sphere.
    """
    
    # Shift image values such that they are positive,
    # since the formula for modulation needs +ve values
    if image.min() <= 0:
        image = image - image.min()

    # Denoise using a median filter
    denoised = median_filter(median_filter(image, 3), 3)
    
    # Save the images containing the gaps
    misc.imsave(folder_name + "touch_img.png", denoised)
    misc.imsave(folder_name + "touch_img.tif", denoised)

    # Calculate the pixel intensities of spheres
    intensity_left = sphere_l_intensity(denoised)
    intensity_right = sphere_r_intensity(denoised)
    
    # This is the modulation at low frequencies, used 
    # to normalize the MTF values between 0% and 100%
    low_freq_left = (intensity_left - np.min(denoised)) /\
                    (intensity_left + np.min(denoised))
    low_freq_right = (intensity_right - np.min(denoised)) /\
                     (intensity_right + np.min(denoised))
    
    gap = []
    mtf_cleft = []
    mtf_cright = []
     
    for i in np.arange(0., image.shape[0] / 2.):
         
        Xdata = []
        Ydata = []
        gapX = []
        gapY = []
        
        distance = dist_between_spheres(r1, r2, i, image.shape[0] / 2.)
        
        signal = [pixel for pixel in denoised[i, :]]
        gap_signal = []
        
        for j in np.arange(0., image.shape[1]):
            # Used to plot line on the iamge
            Xdata.append(j)
            Ydata.append(i)
            
            # If we are in the region where the spheres are separated,
            # stores these vlaues to plot the gap
            if image.shape[1] / 2. + distance / 2. > j > image.shape[1] / 2. - distance / 2.:
                gapX.append(j)
                gapY.append(i)
            
            # Take the region around the gap, which later on will be used
            # to define the intensity at the gap between the spheres.
            if image.shape[1] / 2. + distance > j > image.shape[1] / 2. - distance:
                gap_signal.append(denoised[i, j])
                
             
        if signal: 
            # PLOT THE IMAGE WITH THE LINE ON IT
            if int(i) % 10 == 0:
                pl.imshow(denoised)
                pl.plot(Xdata, Ydata)
                pl.plot(gapX, gapY)
                pl.gray()
                pl.axis('off')
                pl.savefig(folder_name + 'result{0}.png'.format(i))
                pl.close('all')
                
                # PLOT THE LINE PROFILE
                ymax = np.max(denoised)
                ymin = np.min(denoised)
                data = np.array([range(len(signal)), signal]).T
                    
                pl.plot(data[:, 0], data[:, 1])
                pl.ylim(ymin, ymax)
                pl.savefig(folder_name + 'results%i.png' % i)
                pl.close('all')

        # Check if the gap still exists
        if gap_signal:
            # If the signal minima is higher than the minima in the gap
            # it means that the contrast must be lost in the centre - stop
            if np.min(signal) >= np.min(gap_signal):
                
                # Narrow down the region. No point in looking bellow 1 pixels
                # since it can't be resolved anyway
                if distance < 20 and distance >= 1:
                    # Calculate modulation of the signal and normalize
                    gap.append(distance)
                
                    mtf = 100 * modulation(np.min(gap_signal), intensity_left, distance) / low_freq_left
                    mtf_cleft.append(mtf)
                
                    mtf = 100 * modulation(np.min(gap_signal), intensity_right, distance) / low_freq_right
                    mtf_cright.append(mtf)

    ############# LEFT SPHERE #########################
    
    # Consider only values above 9% MTF, since bellow does not
    # matter - can't be resolved anymore
    mtf_resolutionY = [item for item in mtf_cleft if item > 9]
    mtf_resolutionX = [gap[i] for i, item in enumerate(mtf_cleft) if item > 9]
    
    # Fit the data with a curve and get the value of resolution
    best_fit, resolution = fit_data.MTF(mtf_resolutionX, mtf_resolutionY)
    
    pl.gca().invert_xaxis()
    pl.plot(best_fit, mtf_resolutionY, label="best fit")
    pl.plot(mtf_resolutionX, mtf_resolutionY, 'ro', label="left sphere data")
    pl.title("Gap width at 9% (Rayleigh diffraction limit) is {0}".format(resolution))
    pl.xlabel("Width in pixels")
    pl.ylabel("MTF %")
    pl.tight_layout()
    
    save_data(folder_name + 'gap.npy', gap)
    save_data(folder_name + 'best_fit_left.npy', best_fit)
    save_data(folder_name + 'mtf_cleft.npy', mtf_cleft)
    
    pl.savefig(folder_name + 'mtf_left.png')
    pl.close('all')
    
    ############### RIGHT SPHERE #####################
        
    # Consider only values above 9% MTF, since bellow does not
    # matter - can't be resolved anymore
    mtf_resolutionY = [item for item in mtf_cright if item > 9]
    mtf_resolutionX = [gap[i] for i, item in enumerate(mtf_cright) if item > 9]
    
    # Fit the data with a curve and get the value of resolution
    best_fit, resolution = fit_data.MTF(mtf_resolutionX, mtf_resolutionY)
    
    pl.gca().invert_xaxis()
    pl.plot(best_fit, mtf_resolutionY, label="best fit")
    pl.plot(mtf_resolutionX, mtf_resolutionY, 'ro', label="left sphere data")
    pl.title("Gap width at 9% (Rayleigh diffraction limit) is {0}".format(resolution))
    pl.xlabel("Width in pixels")
    pl.ylabel("MTF %")
    pl.tight_layout()
    
    save_data(folder_name + 'gap.npy', gap)
    save_data(folder_name + 'best_fit_right.npy', best_fit)
    save_data(folder_name + 'mtf_cright.npy', mtf_cright)
    
    pl.savefig(folder_name + 'mtf_right.png')
    pl.close('all')

    return


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


def sphere_l_intensity(img):
    """
    Measure pixel intensity of the sphere being
    analysed, which is on the left side of the image
    """
    pixels = []
    for j in range(0, img.shape[0]):
        for i in range(1, 40):
            pixels.append(img[j, i])

    return np.mean(pixels)


def sphere_r_intensity(img):
    """
    Measure pixel intensity of the sphere being
    analysed, which is on the right side of the image
    """
    pixels = []
    for j in range(0, img.shape[0]):
        for i in range(1, 40):
            pixels.append(img[j, img.shape[1] - i])

    return np.mean(pixels)


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


def project_onto_plane(vect):
    """
    Return vector projection onto the xy plane
    """
    x, y, z = vect
    
    return (x, y, 0.)


def find_contact_3D(centroids, radius, tol = 20.):
    """
    Input: Arrays of all the centroids and all radii
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
            print "Difference between radii sum and centre distance is", abs(D - L)
            
            if abs(D - L) <= tol:
                
                touch_pt = vector_3D(c1, c2, r1)
                touch_pts.append(touch_pt)
                centres.append((c1, c2))
                radii.append((r1, r2))
                
    return centres, touch_pts, radii


def sample_rate(P1, P2):
    """
    When we have to loop through pixels at
    an angle, the angle needs to be taken
    into account.
    """
    v = (P1[0] - P2[0], P1[1] - P2[1], P1[2] - P2[2])
    # Project v onto the xy plane
    # xvect is a unit vector on that plane
    normalized = (1. / np.sqrt(2), 1. / np.sqrt(2), 0.)
    
    angle = np.dot(normalized, v) / modulus(v)
    
    # We need 1 / cosA
    return 1. / np.cos(angle)


def get_slice(P1, P2, name, Z):
    """
    Get slice through centre for analysis
    """
    
    centre_dist = distance_3D(P1, P2)
    plot_img = np.zeros((ceil(centre_dist / 2. + 1), centre_dist + 2 ))
    Xrange = np.arange(-centre_dist / 4., centre_dist / 4. + 1)
    
    # time goes along the vector between P1 and P2
    # since it might be at an angle, I can't loop in 1
    # pixel increments - this will miss certain slices. Therefore,
    # I need to loop through by 1/cosA, where A is angle between
    # the xy plane and vector P1->P2
    sampling = sample_rate(P1, P2)
    
    for time in np.linspace(0, centre_dist + 1,
                            centre_dist * sampling):
        # Go up along the line
        new_pt = vector_3D(P1, P2, time)
        old_pt = vector_3D(P1, P2, time - centre_dist / 2. * sampling)

        if time == 0:
            input_file = name % int(round(new_pt[2] + Z, 0))
            img = io.imread(input_file)
            
        # Check if the previous slice is the same as the next
        # don't load it again if it is - save computation time
        if int(round(new_pt[2] + Z, 0)) != int(round(old_pt[2] + Z, 0)):
            
            input_file = name % int(round(new_pt[2] + Z, 0))
            img = io.imread(input_file)
            
            for X in Xrange:
        
                # Get along the X direction for every height
                x, y, z = vector_perpendicular_3D(new_pt, P2, 1, 0, X)
                
                pixel_value = interpolation(x, y, img)
                
                plot_img[X + centre_dist / 4., time] = pixel_value
        else:
            for X in Xrange:
        
                # Get along the X direction for every height
                x, y, z = vector_perpendicular_3D(new_pt, P2, 1, 0, X)

                pixel_value = interpolation(x, y, img)
                
                plot_img[X + centre_dist / 4., time] = pixel_value
                
    return plot_img


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

    pixel_value = 1 / ((x2 - x1) * (y2 - y1)) * (f11 * (x2 - x) * (y2 - y) +
                                                 f21 * (x - x1) * (y2 - y) +
                                                 f12 * (x2 - x) * (y - y1) +
                                                 f22 * (x - x1) * (y - y1))

    return pixel_value


def touch_lines_3D(pt1, pt2, folder_name, name, r1, r2):
    """
    Goes along lines in the region between
    two points.
    Used for obtaining the widths of the gaussian fitted
    to the gap between spheres
    """
    
    # This can vary to take points above or bellow,
    # the point of contact
    Zrange = np.arange(0., 1.)
    
    for Z in Zrange:
        # Create an array to store the slanted image slices
        # used for plotting
        ROI = get_slice(pt1, pt2, name, Z)
        
        pl.imshow(ROI)
        pl.show()
        
        create_dir(folder_name + "plots_%i/" % Z)
        
        fit_and_visualize(ROI, folder_name+ "plots_%i/" % Z, r1, r2)

    return

# import cPickle
# import pylab as pl
#  
# f = open("/home/jjl36382/Presentation data/67/gap.npy", 'r')
# x1 = cPickle.load(f)
# f.close()
# f = open("/home/jjl36382/Presentation data/73/gap.npy", 'r')
# x2 = cPickle.load(f)
# f.close()
# f = open("/home/jjl36382/Presentation data/80/gap.npy", 'r')
# x3 = cPickle.load(f)
# f.close()
# f = open("/home/jjl36382/Presentation data/67/mtf_cleft.npy", 'r')
# y1 = cPickle.load(f)
# f.close()
# f = open("/home/jjl36382/Presentation data/73/mtf_cleft.npy", 'r')
# y2 = cPickle.load(f)
# f.close()
# f = open("/home/jjl36382/Presentation data/80/mtf_cleft.npy", 'r')
# y3 = cPickle.load(f)
# f.close()
# f = open("/home/jjl36382/Presentation data/67/best_fit.npy", 'r')
# b1 = cPickle.load(f)
# f.close()
# f = open("/home/jjl36382/Presentation data/73/best_fit.npy", 'r')
# b2 = cPickle.load(f)
# f.close()
# f = open("/home/jjl36382/Presentation data/80/best_fit.npy", 'r')
# b3 = cPickle.load(f)
# f.close()  
#       
# pl.plot(x1, y1, 'ro', label = "53keV / resolution {0}".format(1.63))
# pl.plot(x3, y3, 'go', label = "75keV / resolution {0}".format(1.06))
# pl.plot(x2, y2, 'bo', label = "130keV / resolution {0}.".format("~2"))
#  
# pl.xlabel("Distance between spheres (pixels)")
# pl.ylabel("MTF %")
# pl.legend()
# pl.gca().invert_xaxis()
#         
# pl.show()


# fit_and_visualize(1,1,382,382)
# p1 = (655.0149163710023, 872.23273666683576, 1410.0824012869405)
# p2 = (766.75647447645542, 875.85630359400147, 793.23713204824833)
# touch_lines_3D(p2, p1, 1, "./67/", "/dls/tmp/tomas_aidukas/scans_july16/cropped/50867/image_%05i.tif", 306.97398172, 307.03693901)