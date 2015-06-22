from PIL import Image
import numpy as np


"""
Inputs:
    Image containing circle/edge
    (NB: Image must be in a lighter background for it to work!
    Otherwise chagne MAX to MIN if it is the opposite!)
    The angle through which the line will be drawn
    Center of the circle

Output:
    The value of the radius obtained from the image
    
This function draws a line from the center and check how that line changes.
At every increment the pixel value along the line is stored.
Then at the point where pixel values suddenly change we can assume that
we reached the edge and at that point we will get the value of the radius.
"""


def get_radius(image, theta, centre):
    
    import pylab as pl
    import math
    import numpy as np
    from skimage.restoration import denoise_tv_chambolle
    
    pl.close('all')
    
    Xc = centre[0]
    Yc = centre[1]
    
    # Denoise the image

    #image = denoise_tv_chambolle(image, weight=0.008)

    # Get values of pixels according to an increasing radius for a same angle
    # Get the radius to be at most half the image since the circle can't be bigger
    R = min(image.shape[0] / 2, image.shape[1] / 2) 
    
    # Simple trig identities
    # R is the max value that we can reach
    delta_x = R * math.sin(theta)
    delta_y = R * math.cos(theta)
     
    points = []
    
    # Go from 0 to 1.001 in steps of 0.001
    for alpha in np.arange(0,1.001,0.001):
        # Xc and Yc are the positions from the center
        # points stores all the points from the center going along the radius
        points.append(image[Xc + alpha * delta_x, Yc + alpha * delta_y])
     
    # Find the radius of the circle
     
    # Calculate discrete difference and find the edge via the extremum
    # diff returns the difference between adjacent points
    dif = np.diff(points)
    # argwhere returns the indices where difference is minimum
    # so if you have an edge then the pixel values of that edge will be small
    # background is white and circles are black.
    # White is 255 pixel value so when we reach the edge and subtract 0 from 255 we will
    # get a MAX value, which will mean that we found the edge.
    index_edge = np.argwhere(dif == np.max(dif))[0][0]
     
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
    
    return radii_circle


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

"""
Test how it works with a clean image.
Use png format - jpg is quite bad
"""

img = np.array(Image.open("segmented_real.png").convert('L'))
cent = (184,184)

"""
The spikes in the radius at a few angles are probably due to the pixelated
nature of the circle, which means that it misses the edge pixel.
This is why I should use a filled circle
"""
rad, rad_flat = remove_large_sine(img, cent)

print rad
print rad_flat
