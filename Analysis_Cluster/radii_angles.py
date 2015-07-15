def get_radius(image_area, theta, phi, centre, rad_min):
    
    import pylab as pl
    import numpy as np
    from skimage.restoration import denoise_tv_chambolle
    from scipy.ndimage import median_filter,gaussian_filter
    from scipy.ndimage.filters import gaussian_filter1d
    pl.close('all')
    
    # Plot value of pixel as a function of radius
    
    
    Xc = centre[0]
    Yc = centre[1]
    Zc = centre[2]
    
    R = min(image_area.shape[0] / 2, image_area.shape[1] / 2, image_area.shape[2] / 2) - 1
    
    delta_x = R * np.sin(phi) * np.cos(theta)
    delta_y = R * np.sin(phi) * np.sin(theta)
    delta_z = R * np.cos(phi)
    
    points = []
    
    step = 0.001
    saved_alpha = []
    #rad_min = 250
    
    for alpha in np.arange(0,1+step,step):
        if np.sqrt((alpha*delta_x)**2 + (alpha*delta_y)**2 + (alpha*delta_z)**2) > rad_min:
            saved_alpha.append(alpha)
            points.append( image_area[int(Xc + alpha * delta_x), int(Yc + alpha * delta_y), int(Zc + alpha * delta_z)] )
    
    # Find the radius of the circle
    
    # Calculate discrete difference and find the edge via the extremum
    #points = gaussian_filter1d(np.asarray(points), 15)
    dif = np.diff(points)
    
    
    index_edge = np.argwhere(dif == np.min(dif))[0][0]
    
    # Calculate the radius
    radius_sphere = index_edge*step*R + rad_min
    
    print radius_sphere
    
    return round(radius_sphere, 2)

def plot_radii(image_area, centre, start, stop):
    
    import numpy as np
    import pylab as pl
    from skimage.filter import threshold_otsu


    pl.close('all')
    
    # threshold
    threshold = threshold_otsu(image_area, 2)
    image_area = (image_area >= threshold) * 1

    # Calculate radii for every angle 
    
    step = 1
    theta_bord = np.arange(start,stop,step)
    phi_bord = np.arange(0,180,step)
    
    radii_sphere = np.zeros( (len(theta_bord), len(phi_bord)) )
    
#     radii_sphere[0, 0] = get_radius(image_area, 0, 0, centre, 0)
#     rad_min = radii_sphere[0, 0] - (image_area.shape[2]/2 - radii_sphere[0, 0])
#     
    rad_min = 0
    for theta in theta_bord:
        for phi in phi_bord:
            theta_rad = np.radians(theta)
            phi_rad = np.radians(phi)
            radii_sphere[(theta-start)/step, phi/step] = get_radius(image_area, theta_rad, phi_rad, centre, rad_min)
    
    return radii_sphere


# import numpy as np
# import pylab as pl
#     
# image_area = np.load("/dls/tmp/jjl36382/complicated_data/spheres/spheresobel1.npy")[:,:,:]
#      
# # pl.imshow(image_area)
# # pl.show()
#      
# print image_area.shape[0]
# print image_area.shape[1]
# print image_area.shape[2]
#      
# centre = (int(128 * 1.2), int(128 * 1.2), int(128 * 1.2))
# #centre = (380*1.2,380*1.2,380*1.2)
# start = 0
# stop = 359
# step = 10
#     
# plot_radii(image_area, centre, start, stop)