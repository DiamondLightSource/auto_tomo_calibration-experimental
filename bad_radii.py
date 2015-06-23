def get_radius(image_area, theta, phi, centre):
     
    import pylab as pl
    import numpy as np
     
    pl.close('all')
     
    # Plot value of pixel as a function of radius
     
    Xc = centre[0]
    Yc = centre[1]
    Zc = centre[2]
     
    R = min(image_area[0].shape[0] / 2, image_area[0].shape[1] / 2, len(image_area) / 2)
     
    delta_x = R * np.sin(phi) * np.cos(theta)
    delta_y = R * np.sin(phi) * np.sin(theta)
    delta_z = R * np.cos(phi)
     
    points = []
    for alpha in np.arange(0,1.001,0.001):
        points.append( image_area [int(Zc + alpha * delta_z)] [int(Xc + alpha * delta_x), int(Yc + alpha * delta_y)] )
     
    # Find the radius of the circle
     
    # Calculate discrete difference and find the edge via the extremum
    dif = np.diff(points)
     
    index_edge = np.argwhere(dif == np.min(dif))[0][0]
     
    # Calculate the radius
    radius_sphere = index_edge*0.001*R
     
    return round(radius_sphere, 2)
 
 
def plot_radii(image_area, radius, centre):
     
    import numpy as np
    import pylab as pl
     
    pl.close('all')
     
    # Calculate radii for every angle 
     
    theta_bord = np.arange(0,360,5)
    phi_bord = np.arange(0,180,5)
     
    radii_sphere = np.ones( (len(theta_bord), len(phi_bord)) )
     
    for theta in theta_bord:
        print theta
        for phi in phi_bord:
            theta_rad = np.radians(theta)
            phi_rad = np.radians(phi)
            radii_sphere[theta/5, phi/5] = get_radius(image_area, theta_rad, phi_rad, centre)
     
    # Plot
     
    #min_value = 20
    #max_value = 40
     
    pl.imshow(radii_sphere.T)#, vmin=min_value, vmax=max_value)
    pl.title(r'Radii of sphere as a function of 2 spherical angles $\theta$ and $\phi$', fontdict={'fontsize': 16,'verticalalignment': 'bottom','horizontalalignment': 'center'})
    pl.xlabel(r'$\theta$', fontdict={'fontsize': 14,'verticalalignment': 'top','horizontalalignment': 'center'})
    pl.ylabel(r'$\phi$', fontdict={'fontsize': 14,'verticalalignment': 'bottom','horizontalalignment': 'right'}, rotation=0)
    pl.colorbar(shrink=0.8)#, boundaries=[min_value,max_value])
     
    pl.show()
     
    return radii_sphere


import numpy as np

img_3d = np.load("img_3d.npy")
cent_3d = np.load("cent_3d.npy")
rad_3d = np.load("rad_3d.npy")

img_3d = np.asarray(img_3d)[0]

#int_3d = 1*img_3d

print img_3d[5]
plot_radii(img_3d, rad_3d[0], cent_3d[0])