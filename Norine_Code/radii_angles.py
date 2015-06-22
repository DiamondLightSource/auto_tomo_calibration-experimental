def get_radius(image, theta, centre):
    
    import pylab as pl
    import math
    import numpy as np
    
    pl.close('all')
    
    # Plot value of pixel as a function of radius
    
    Xc = centre[0]
    Yc = centre[1]
    
    R = min(image.shape[0] / 2, image.shape[1] / 2)
    print R
    
    delta_x = R * math.sin(theta)
    delta_y = R * math.cos(theta)
    
    points = []
    
    for alpha in np.arange(0,1.001,0.001):
        points.append(image[int(Xc + alpha * delta_x), int(Yc + alpha * delta_y)])
    
    # Find the radius of the circle
    
    # Calculate discrete difference and find the edge via the extremum
    dif = np.diff(points)
    pl.plot(np.arange(0,1,0.001)*R, dif)
    pl.xlabel('radius')
    pl.ylabel('discrete difference')
    pl.show()
    index_edge = np.argwhere(dif == np.min(dif))[0][0]
    
    # Calculate the radius
    radius_circle = index_edge*0.001*R
    
    # Plot
    
    pl.plot(np.arange(0,1.001,0.001)*R, points)
    pl.xlabel('radius')
    pl.ylabel('value of pixel')
    # Plot annotations
    pl.plot([0, radius_circle], [points[index_edge], points[index_edge]], color='green', linewidth=2, linestyle="--")
    pl.plot([radius_circle, radius_circle], [-0.001, points[index_edge]], color='green', linewidth=2, linestyle="--")
    pl.annotate('radius', xy=(radius_circle, -0.001), xytext=(+10, +40), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->"))
    pl.show()
    
    return round(radius_circle, 2)

def plot_radii(image, radius, centre):
    
    import numpy as np
    import pylab as pl
    import math
    from scipy import interpolate
    
    pl.close('all')
    
    # Calculate radii for every angle 
    
    radii_circle = []
    
    theta_bord = np.arange(0,360,1)
    
    for theta in theta_bord:
        theta_pi = (math.pi * theta) / 180
        radii_circle.append(get_radius(image, theta_pi, centre[0], centre[1]))
    
    # Smooth the curve
    
    tck = interpolate.splrep(theta_bord, radii_circle, s=450)
    radii_new = interpolate.splev(np.arange(0,360,0.1), tck, der=0)
    #pl.plot(np.arange(0,360,0.1), radii_new, '-')
    
    # Plot
    
    pl.plot(theta_bord, radii_circle)
    pl.title('Radii(angles) for model - irregular circle')
    pl.xlabel('angle')
    pl.ylabel('radius')
    #pl.ylim(radius - 20, radius + 20)
    pl.ylim(312,330)
    pl.xlim(0,360)
    pl.show()
    
    return radii_circle

def f(x, A, phi, offset, dilation_coef): # Sine wave to remove from data
    
    import numpy as np
    
    x_rad = x * np.pi / 180
    value = A * np.sin((x_rad - phi) / dilation_coef) + offset
    
    return value

def remove_large_sine(image, radius, centre):
    
    import pylab as pl
    from scipy.optimize import curve_fit
    from scipy import interpolate
    
    # Get radii
    
    theta_bord = np.arange(0,360,1)
    radii_circle = plot_radii(image, radius, centre)
    
    # Fit the curve
    
    popt, pcov = curve_fit(f, theta_bord, radii_circle)
    print 'Parameters of large sine (A, phi, offset, dilation_coef):'
    print popt[0], '   ', popt[1] * 180 / np.pi, '   ', popt[2], '   ', popt[3]
    
    A = popt[0]
    phi = popt[1]
    offset = popt[2]
    dilation_coef = popt[3]
    '''
    # Plot
    
    pl.plot(theta_bord, radii_circle, 'g', theta_bord, f(theta_bord, A, phi, offset, dilation_coef), 'r')
    pl.xlabel('angle')
    pl.ylabel('radius')
    pl.title('Large sine wave')
    pl.xlim(0,360)
    pl.ylim(300,340)
    pl.show()
    '''
    # Flatten data
    
    # Remove the large sine wave from data
    radii_flattened = radii_circle - f(theta_bord, A, phi, offset, dilation_coef) + offset
    '''
    # Smooth the curve
    tck = interpolate.splrep(theta_bord, radii_flattened, s=400)
    radii_new = interpolate.splev(np.arange(0,360,0.1), tck, der=0)
    pl.plot(np.arange(0,360,0.1), radii_new, 'b', label='Smoothed curve')
    '''
    # Plot
    
    pl.plot(theta_bord, radii_flattened)
    pl.xlabel('angle')
    pl.ylabel('radius')
    pl.title('Radii flattened')
    pl.xlim(0,360)
    pl.ylim(313,330)
    pl.show()
    
    return