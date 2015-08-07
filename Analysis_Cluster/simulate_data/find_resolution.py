import numpy as np
import pylab as pl
from scipy import optimize
from math import isnan
from scipy.ndimage.filters import gaussian_filter
from skimage import io
import os

#################################### Fitting fns ############################


def param_est(points):
    from lmfit import Model
    
    points = gaussian_filter(points, 1)
    data = np.array([range(len(points)), points]).T

    x = data[:, 0]
    y = data[:, 1]
    
    gmod = Model(gaussian)
        # find the initial guesses
    prob = data[:, 1] / data[:, 1].sum()  # probabilities
    mu = prob.dot(data[:, 0])  # mean or location param.
    sigma = np.sqrt(prob.dot(data[:, 0] ** 2) - mu ** 2)
#     mu = sum(data[:, 0]*data[:, 1])/sum(data[:, 1])
#     sigma = np.sqrt(abs(sum((data[:, 0]-mu)**2*data[:, 1])/sum(data[:, 1])))
    
    if isnan(sigma):
        print "fitting failed"
        return False, False
    else:
        try:
            centre_guess = mu
        except:
            centre_guess = len(points) / 2.
        
        try:
            height_guess = data[:,1][centre_guess]#np.argwhere(points == np.min(points))[0][0]
        except:
            height_guess = 1.
                
        sigma_guess = sigma        
        # the initial guesses for the Gaussians
        guess1 = [-abs(height_guess), centre_guess, sigma_guess, abs(height_guess)]
        
    height_guess, centre_guess, sigma_guess, ayy = guess1
    
    result = gmod.fit(y, x=x, height=height_guess, center=centre_guess, width=sigma_guess, offset=ayy)
    
    #print(result.fit_report())

    pl.subplot(1, 3, 2)
    pl.plot(x, y)
    pl.plot(x, result.best_fit, 'r-', label = "gaussian filter")
    pl.legend()
    
#     pl.subplot(1, 2, 2)
#     pl.plot(data[:,0], data[:,1], label="true data")
#     pl.plot(data[:,0], gaussianz(data[:,0], *p), label="fitted gaussian")
#     #pl.title("gaussian width %f" % abs(p[2]))
#     pl.legend()
    
    return

def gaussianz(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-cen)**2 /(2*wid**2))


def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset


def create_dir(directory):
    
    if not os.path.exists(directory):
        os.makedirs(directory)


def fit_gaussian_to_signal(points, touch_pt):
    """
    Detects peaks
    First guess for the Gaussian is the firs maxima in the signa;
    """
    
    import pywt
    param_est(points)
    
    # perform discrete wavelet transform
    points, cD = pywt.dwt(points, 'db1')#, 'sp1')
    
    # make the array into the correct format
    data = np.array([range(len(points)), points]).T
    
    # find the initial guesses
    prob = data[:, 1] / data[:, 1].sum()  # probabilities
    mu = prob.dot(data[:, 0])  # mean or location param.
    sigma = np.sqrt(prob.dot(data[:, 0] ** 2) - mu ** 2)
#     mu = sum(data[:, 0]*data[:, 1])/sum(data[:, 1])
#     sigma = np.sqrt(abs(sum((data[:, 0]-mu)**2*data[:, 1])/sum(data[:, 1])))
    
    if isnan(sigma):
        print "fitting failed"
        return False, False
    else:
        try:
            centre_guess = mu
        except:
            centre_guess = len(points) / 2.
        
        try:
            height_guess = data[:,1][centre_guess]#np.argwhere(points == np.min(points))[0][0]
        except:
            height_guess = 1.
                
        sigma_guess = sigma        
        # the initial guesses for the Gaussians
        guess1 = [-abs(height_guess), centre_guess, sigma_guess, abs(height_guess)]

        
        # the functions to be minimized
        errfunc1 = lambda p, xdata, ydata: (gaussian(xdata, *p) - ydata)
     
        # TODO: PCOV SQRT GIVES THE ERRORS FOR THE STDEV
        p, pcov = optimize.leastsq(errfunc1, guess1[:],
                                    args=(data[:,0], data[:,1]))
        
        
        ######################## Plot the fit #############################
        pl.subplot(1, 3, 3)
        pl.plot(data[:,0], data[:,1], label="true data")
        pl.plot(data[:,0], gaussian(data[:,0], *p), label="cwt filter")
        pl.title("gaussian width %f" % abs(p[2]))
        pl.legend()
        
        
        return p, sigma


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


def thresh_MAD(signal):
    """
    Remove outliers using MAD
    """
    mad = MAD(signal)
    M = np.median(np.sort(signal))
    up_thresh = M + mad
    bot_thresh = M - mad
    
    clean_signal = []
    
    for i in signal:
        if i <= up_thresh and i >= bot_thresh:
            clean_signal.append(i)
    
    return clean_signal


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


def plot_line(image, pt):
    """
    pt is the point to be marked in the image
    """
    pl.imshow()
    pl.gray()
    pl.show()
    


def touch_lines_3D(pt1, pt2, image, sampling, touch_pt, plots):
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
    
    min_widths = []
    mean_widths = []
    median_widths = []
    total_widths = []
    
    for Z in Zrange:
        widths = []
        
        for X in Xrange:
            
            # plot line on the image copy
            plot_img = image.copy()
            # Draw a line parallel to the one through the 
            # point of contact at height Z
            P1 = vector_perpendicular_3D(pt1, pt2, 1, Z, X)
            P2 = vector_perpendicular_3D(pt1, pt2, 2, Z, X)
            
            line = []
            length = distance_3D(P1, P2)
            
            # Create an array to store the slanted image slices
            # used for plotting
            plot_image = np.zeros_like((length, length))

            # go along that line
            for time in np.linspace(0., length + 1, length*sampling):
                try:
                    # line coordinates going through the gap
                    x, y, z = vector_3D(P1, P2, time)
                    x, y, z = (int(round(x,0)), int(round(y,0)), int(round(z,0)))
                    
                    pixel_value = image[x, y, z]
                    line.append(pixel_value)
                     
                    ######################## Visualizing the line plotting ######################
                    #print "coordinates?", int(round(X + touch_pt,0)), int(round(time,0))
                    plot_image[int(round(X + touch_pt,0)), int(round(time,0))] = pixel_value
                    plot_img[x, y, z] = max(image.flatten())
                    plot_img[touch_pt, touch_pt, int(round(z,0))] = 0.
#                     pl.imshow(plot_img[:,:,int(round(z,0))])
#                     pl.gray()
#                     pl.pause(0.001)
                    #############################################################################
                    
                except:
                    continue
            pl.imshow(plot_image)
            pl.show()
            # if it is not empty
            if line:

                # Best Gaussian fit
                parameters, sigma = fit_gaussian_to_signal(line, touch_pt)
                
                ###### PLOT GAUSSIAN + ITS LINE ON THE IMAGE ###########
                pl.subplot(1, 3, 1)
                pl.imshow(plot_img[:,:,int(round(z,0))])
                pl.gray()
                
                # Create a folder
                folder_name = plots + "gauss_vs_line_%i/" % (Z + touch_pt)
                create_dir(folder_name)
                pl.savefig(folder_name + 'plots_%i.png' % (X + touch_pt))
                pl.close('all')
                
                ##########################################################
                
                # find the smallest standard deviation
                # Should be at near the touch point
                if sigma != False:

                    widths.append(abs(parameters[2])/sampling)
                    total_widths.append(abs(parameters[2])/sampling)
                else:
                    print x, y, z

        if widths:
            # clean all the outliers (very large widths)
            cleaned_widths = thresh_MAD(thresh_MAD(widths))
            cleaned_total = thresh_MAD(thresh_MAD(total_widths))
            
            # get the minimum, mean and the median width
            # for each slice
            min_widths.append(np.min(cleaned_widths))
            mean_widths.append(np.mean(cleaned_widths))
            median_widths.append(np.median(np.sort(cleaned_widths)))
    
    print ""
    print "GETS A SIGNAL ALONG A LINE AROUND THE POINT OF CONTACT"
    print "STORES THE GAUSSIAN WIDTH OF THAT SIGNAL"
    print "DOES THIS FOR THE WHOLE PLANE"
    print "AND THEN TAKES MINIMUM / MEAN / MEDIAN WIDTH OF THE WHOLE SLICE"
    print "THIS IS REPEATED FOR DIFFERENT HEIGHTS AND AN ARRAY OF"
    print "MINIMUM / MEAN / MEDIAN WIDTHS IS CREATED"
    print ""
    print "ARRAY WITH MIN WIDTHS OF EACH SLICE"
    print "min", np.min(min_widths)
    print "mean", np.mean(min_widths)
    print "median", np.median(np.sort(min_widths))
    print ""
    
    print "ARRAY WITH MEAN WIDTHS OF EACH SLICE"
    print "min sigma", np.min(mean_widths)
    print "mean sigma", np.mean(mean_widths)
    print "median sigma", np.median(np.sort(mean_widths))
    print ""
    
    print "ARRAY WITH MEDIAN WIDTHS OF EACH SLICE"
    print "min sigma", np.min(median_widths)
    print "mean sigma", np.mean(median_widths)
    print "median sigma", np.median(np.sort(median_widths))
    print ""

    print "THESE ARE ALL OF THE WIDTHS OF ALL THE SIGNALS"
    print "AT EVERY HEIGHT"
    print ""
    print "ARRAY WITH WIDTHS OF ALL THE AREA"
    print "min sigma", np.min(cleaned_total)
    print "mean sigma", np.mean(cleaned_total)
    print "median sigma", np.median(np.sort(cleaned_total))
    print ""


    return np.mean(cleaned_total)
