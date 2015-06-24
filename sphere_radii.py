def get_radius(image_area, theta, phi, centre, rad_min):
    
    import pylab as pl
    import numpy as np
    
    pl.close('all')
    
    # Plot value of pixel as a function of radius
    
    Xc = centre[0]
    Yc = centre[1]
    Zc = centre[2]
    
    #R = min(image_area[0].shape[0] / 2, image_area[0].shape[1] / 2, len(image_area) / 2) - 1 #if slices of images in list
    R = min(image_area.shape[0] / 2, image_area.shape[1] / 2, image_area.shape[2] / 2) - 1
    
    delta_x = R * np.sin(phi) * np.cos(theta)
    delta_y = R * np.sin(phi) * np.sin(theta)
    delta_z = R * np.cos(phi)
    
    points = []
    
    step = 0.001
    saved_alpha = []
    
    for alpha in np.arange(0,1+step,step):
        #points.append( image_area [int(Zc + alpha * delta_z)] [int(Xc + alpha * delta_x), int(Yc + alpha * delta_y)] ) #if slices of images in list
        if np.sqrt((alpha*delta_x)**2 + (alpha*delta_y)**2 + (alpha*delta_z)**2) > rad_min:
            saved_alpha.append(alpha)
            points.append( image_area[int(Xc + alpha * delta_x), int(Yc + alpha * delta_y), int(Zc + alpha * delta_z)] )
    
    # Find the radius of the circle
    
    # Calculate discrete difference and find the edge via the extremum
    dif = np.diff(points)
    #pl.plot(np.arange(0,1-step,step)*R, dif)
    #pl.xlabel('radius')
    #pl.ylabel('discrete difference')
    #pl.show()
    index_edge = np.argwhere(dif == np.min(dif))[0][0]
    
    # Calculate the radius
    radius_sphere = index_edge*step*R + rad_min
    
    # Plot
    
    #pl.plot([x*R for x in saved_alpha], points)
    #pl.xlabel('radius')
    #pl.ylabel('value of pixel')
    # Plot annotations
    #pl.plot([0, radius_sphere], [points[index_edge], points[index_edge]], color='green', linewidth=2, linestyle="--")
    #pl.plot([radius_sphere, radius_sphere], [-0.001, points[index_edge]], color='green', linewidth=2, linestyle="--")
    #pl.annotate('radius', xy=(radius_sphere, -0.001), xytext=(+10, +40), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->"))
    #pl.show()
    
    return round(radius_sphere, 2)

def plot_radii(image_area, radius, centre):
    
    import numpy as np
    import pylab as pl
    from scipy import interpolate
    
    pl.close('all')
    
    # Calculate radii for every angle 
    
    step = 20
    start = 0
    theta_bord = np.arange(start,360+step,step)
    phi_bord = np.arange(start,180+step,step)
    
    radii_sphere = np.zeros( (len(theta_bord), len(phi_bord)) )
    
    radii_sphere[0, 0] = get_radius(image_area, 0, 0, centre, 0)
    rad_min = radii_sphere[0, 0] - (image_area.shape[2]/2 - radii_sphere[0, 0]) 
    
    print 'Theta = '
    for theta in theta_bord:
        print theta,
        for phi in phi_bord:
            theta_rad = np.radians(theta)
            phi_rad = np.radians(phi)
            radii_sphere[(theta-start)/step, (phi-start)/step] = get_radius(image_area, theta_rad, phi_rad, centre, rad_min)
    
    # Plot
    
    min_value = 20
    max_value = 40
    
    pl.imshow(radii_sphere.T)
    pl.title(r'Radii of sphere as a function of 2 spherical angles $\theta$ and $\phi$',\
             fontdict={'fontsize': 16,'verticalalignment': 'bottom','horizontalalignment': 'center'})
    pl.xlabel(r'$\theta$', fontdict={'fontsize': 14,'verticalalignment': 'top','horizontalalignment': 'center'})
    pl.ylabel(r'$\phi$', fontdict={'fontsize': 14,'verticalalignment': 'bottom','horizontalalignment': 'right'}, rotation=0)
    #pl.xticks(np.arange(0, len(theta_bord)+1, 10), theta_bord)
    #pl.yticks(np.arange(0, len(phi_bord)+1, 10), phi_bord)
    pl.colorbar(shrink=0.8)
    
    pl.show()
    
    return

def f(x, A, phi, offset, dilation_coef): # Sine wave to remove from data
    
    import numpy as np
    
    x_rad = np.radians(x)
    value = A * np.sin((x_rad - phi) / dilation_coef) + offset
    
    return value

def remove_large_sine(image_area, radius, centre):
    
    import numpy as np
    import pylab as pl
    from scipy.optimize import curve_fit
    from scipy import interpolate
    
    # Get radii
    
    theta_bord = np.arange(0,360,1)
    radii_sphere = plot_radii(image_area, radius, centre)
    
    # Fit the curve
    
    popt, pcov = curve_fit(f, theta_bord, radii_sphere)
    print 'Parameters of large sine (A, phi, offset, dilation_coef):'
    print popt[0], '   ', np.degrees(popt[1]), '   ', popt[2], '   ', popt[3]
    
    A = popt[0]
    phi = popt[1]
    offset = popt[2]
    dilation_coef = popt[3]
    '''
    # Plot
    
    pl.plot(theta_bord, radii_sphere, 'g', theta_bord, f(theta_bord, A, phi, offset, dilation_coef), 'r')
    pl.xlabel('angle')
    pl.ylabel('radius')
    pl.title('Large sine wave')
    pl.xlim(0,360)
    pl.ylim(300,340)
    pl.show()
    '''
    # Flatten data
    
    # Remove the large sine wave from data
    radii_flattened = radii_sphere - f(theta_bord, A, phi, offset, dilation_coef) + offset
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


import numpy as np

recon = np.load("reconstructed.npy")
cent_3d = np.load("cent_3d.npy")
rad_3d = np.load("rad_3d.npy")
orig = np.load("original_3d.npy")
sphere_crops = np.load("crops_3d.npy")

#img_3d = np.asarray(img_3d)[0]

#crop_one = sphere_crops[0]
#orig_one_sphere = orig[0]
#print crop_one
#print orig.shape

print recon[0].shape
print orig.shape

plot_radii(recon[0], rad_3d[0], cent_3d[0])
#plot_radii(img_3d,30,(50,50,50))