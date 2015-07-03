from PIL import Image
import numpy as np


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


def get_radius(image, theta, centre):
    
    import pylab as pl
    import math
    import numpy as np
    from scipy.ndimage.filters import gaussian_filter
    from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
    from scipy.signal import medfilt2d

    pl.close('all')
    
    Xc = centre[0]
    Yc = centre[1]
    
    # Denoise the image
    
    #image = denoise_tv_chambolle(image, weight=0.5)
    #pl.imsave("denoised.png",image)

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
        # Xc and Yc are the positions from the center
        # points stores all the points from the center going along the radius
        points.append(image[Xc + alpha * delta_x, Yc + alpha * delta_y])
    
    #points = smooth(np.asarray(points), 3)
    # Find the radius of the circle
    #print points 
    # Calculate discrete difference and find the edge via the extremum
    # diff returns the difference between adjacent points
    dif = np.diff(points)
    # argwhere returns the indices where difference is minimum
    # so if you have an edge then the pixel values of that edge will be small
    # background is white and circles are black.
    # White is 255 pixel value so when we reach the edge and subtract 0 from 255 we will
    # get a MAX value, which will mean that we found the edge.
    index_edge = np.argwhere(dif == np.max(dif))[0][0]
    #index_edge = dif.argmax(axis=0) 
    # Calculate the radius
    # R was the max length of our line
    # index_edge gives * 0.001 the fraction of R where the edge is
    # so we multiply by R as well and get the radius
    radius_circle = index_edge*0.001*R
     
    # Plot
    """
    # distance from center vs pixel values
    pl.plot(np.arange(0,1.001,0.001)*R, points)
    pl.xlabel('radius')
    pl.ylabel('value of pixel')
    # Plot annotations
    pl.plot([0, radius_circle], [points[index_edge], points[index_edge]], color='green', linewidth=2, linestyle="--")
    pl.plot([radius_circle, radius_circle], [-0.001, points[index_edge]], color='green', linewidth=2, linestyle="--")
    pl.annotate('radius', xy=(radius_circle, -0.001), xytext=(+10, +40), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->"))
    pl.show()"""
     
    return radius_circle



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


def plot_radii(image, centre):
    
    import numpy as np
    import pylab as pl
    import math
    from scipy import interpolate
    
    pl.close('all')
    
    pl.imshow(image)
    pl.show()
    
    print "after padding..."
    print "center is..", centre[0]+10, centre[1]+10
    image = np.pad(image,10, 'edge')
    pl.imshow(image)
    pl.show()
    
    centre = centre[0] + 10, centre[1] + 10
    
    # Calculate radii for every angle
    radii_circle = []
    
    theta_bord = np.arange(0, 360, 1)
    
    for theta in theta_bord:
        theta_pi = (math.pi * theta) / 180.0
        radii_circle.append(get_radius(image, theta_pi, centre))
    
    # Smooth the curve
    
    """
    tck = interpolate.splrep(theta_bord, radii_circle, s=450)
    radii_new = interpolate.splev(np.arange(0,360,1), tck, der=0)
    pl.plot(np.arange(0,360,1), radii_new, '-')
    #pl.show()
    """

    # Plot
    radius = np.mean(radii_circle)
    
    pl.plot(theta_bord, radii_circle)
    pl.title('Radii(angles) for model - irregular circle')
    pl.xlabel('angle')
    pl.ylabel('radius')
    #pl.ylim(radius-5,radius+5)
    pl.xlim(0,360)
    pl.show()
    
    return np.mean(radii_circle)


"""
Creates a sine curve to fit the displaced circles
"""


def f(x, A, phi, offset, dilation_coef):
    
    import numpy as np
    
    x_rad = x * np.pi / 180
    value = A * np.sin((x_rad - phi) / dilation_coef) + offset
    
    return value

"""
This sine wave removal acts as a correction to the displaced center.

However, if the centre is already at a correct position and you get an almost
straight line then this correction distorts the result!!
"""


def remove_large_sine(image, centre):
    
    import pylab as pl
    from scipy.optimize import curve_fit
    from scipy import interpolate
    
    # Get radii
    
    theta_bord = np.arange(0, 360, 1)
    radii_circle = plot_radii(image, centre)
    
    # Fit the curve
    
    popt, pcov = curve_fit(f, theta_bord, radii_circle)
    print 'Parameters of large sine (A, phi, offset, dilation_coef):'
    print popt[0], '   ', popt[1] * 180 / np.pi, '   ', popt[2], '   ', popt[3]
    
    A = popt[0]
    phi = popt[1]
    offset = popt[2]
    dilation_coef = popt[3]
    
    # Mean radius
    radius = np.mean(radii_circle)
    
    # Sine data
    sine = f(theta_bord, A, phi, offset, dilation_coef)
    # Plot
    
    pl.plot(theta_bord, radii_circle, 'g', theta_bord, sine, 'r')
    pl.xlabel('angle')
    pl.ylabel('radius')
    pl.title('Large sine wave')
    pl.xlim(0, 360)
    # pl.ylim(radius-5,radius+5)
    pl.show()
    
    # Remove the large sine wave from data
    radii_flattened = radii_circle - sine + offset
    '''
    # Smooth the curve
    tck = interpolate.splrep(theta_bord, radii_flattened, s=400)
    radii_new = interpolate.splev(np.arange(0,360,0.1), tck, der=0)
    pl.plot(np.arange(0,360,0.1), radii_new, 'b', label='Smoothed curve')
    '''
    # Plot
    radius_flat = np.mean(radii_flattened)
    
    pl.plot(theta_bord, radii_flattened)
    pl.xlabel('angle')
    pl.ylabel('radius')
    pl.title('Radii flattened')
    pl.xlim(0, 360)
    # pl.ylim(radius-5,radius+5)
    pl.show()
    
    return [radius, radius_flat]



def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    import numpy
    
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y
