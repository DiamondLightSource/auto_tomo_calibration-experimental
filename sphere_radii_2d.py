def get_radius(image_area, theta, phi, centre,):
    
    import pylab as pl
    import numpy as np
    
    pl.close('all')
    
    # Plot value of pixel as a function of radius
    
    Xc = centre[0]
    Yc = centre[1]
    Zc = centre[2]
    
    R = min(image_area[0].shape[0] / 2, image_area[0].shape[1] / 2, len(image_area) / 2) - 1 #if slices of images in list
    #R = min(image_area.shape[0] / 2, image_area.shape[1] / 2, image_area.shape[2] / 2) - 1
    
    delta_x = R * np.sin(phi) * np.cos(theta)
    delta_y = R * np.sin(phi) * np.sin(theta)
    delta_z = R * np.cos(phi)
    
    points = []
    
    step = 0.001
    saved_alpha = []
    
    for alpha in np.arange(0,1+step,step):
        points.append( image_area [int(Zc + alpha * delta_z)] [int(Xc + alpha * delta_x), int(Yc + alpha * delta_y)] ) #if slices of images in list
        """if np.sqrt((alpha*delta_x)**2 + (alpha*delta_y)**2 + (alpha*delta_z)**2) > rad_min:
            saved_alpha.append(alpha)
            points.append( image_area [int(Zc + alpha * delta_z)] [int(Xc + alpha * delta_x), int(Yc + alpha * delta_y)] )
    """
    # Find the radius of the circle
    
    # Calculate discrete difference and find the edge via the extremum
    dif = np.diff(points)
    #pl.plot(np.arange(0,1-step,step)*R, dif)
    #pl.xlabel('radius')
    #pl.ylabel('discrete difference')
    #pl.show()
    index_edge = np.argwhere(dif == np.min(dif))[0][0]
    
    # Calculate the radius
    radius_sphere = index_edge*step*R
    
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
    
    step = 5
    start = 0
    theta_bord = np.arange(start,360+step,step)
    phi_bord = np.arange(start,180+step,step)
    
    radii_sphere = np.zeros( (len(theta_bord), len(phi_bord)) )
    
    print 'Theta = '
    for theta in theta_bord:
        print theta,
        for phi in phi_bord:
            theta_rad = np.radians(theta)
            phi_rad = np.radians(phi)
            radii_sphere[(theta-start)/step, (phi-start)/step] = get_radius(image_area, theta_rad, phi_rad, centre)
    
    # Plot
    
    #min_value = 20
    #max_value = 40
    
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


import numpy as np
import pylab as pl

# Load generated data
recon = np.load("./Numpy_Files/reconstructed.npy")
cent_3d = np.load("./Numpy_Files/cent_3d.npy")
rad_3d = np.load("./Numpy_Files/rad_3d.npy")
orig = np.load("./Numpy_Files/original_3d.npy")

plot_radii(recon[0], rad_3d[0], cent_3d[0])