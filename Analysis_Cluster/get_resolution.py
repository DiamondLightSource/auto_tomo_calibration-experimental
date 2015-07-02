def load_data(path):
    """
    Load the files in the directory and returns
    them as arrays
    
    Input:
        path string
        
    Output:
        radii np array and outliers dictionary
    """
    import os
    import numpy as np
    import pickle
    
    # Read in the filenames from the directory
    radii = []
    outliers = []
    for f in os.listdir(path):
        if f.endswith(".npy") and f.startswith("radii"):
            radii.append(os.path.join(path, f))
        elif f.endswith(".dat") and f.startswith("outliers"):
            outliers.append(os.path.join(path, f))
    
    radii_names = sorted(radii)
    dict_names = sorted(outliers)
    
    # a list of radii vs angle
    radii_spheres = []
    # this is a list of dicts
    angles_outliers = []
    for i in range(len(radii_names)):
        radii_spheres.append(np.load(radii_names[i]))
        
        f = open(dict_names[i], 'r')
        angles_outliers.append(pickle.load(f))
        f.close()
    
    return radii_spheres, angles_outliers


def approx_value(x, y, value, tolerance=5):
    """
    Check if two values are within a certain tolerance
    
    Input:
        two values and a tolerance
    
    Output:
        True boolean value
    """
    if (value - tolerance) <= (x + y) <= (value + tolerance):
        return True


def find_contact():
    """
    Loads up the dictionary containing the outlier radii
    and their angles. This will load up all the outlier arrays
    for all of the spheres.
    
    If the radii are known of the spheres and they are touching
    then we need to calculate the touching point using the outlier data.
    
    The angle difference must equate to 180 degrees for alpha
    and the sum of phi to 180. This then means
    that at that point the spheres are touching each other.
    Radii not satisfying this condition are probably just displaced
    due to the inaccurate center position
    
    Once the point of contact is known, then the function will go along one
    direction and check at what angle the radii become normal. Using this angle
    for both spheres which are touching it is possible to calculate the
    distance where they can be resolved. This distance is the resolution.
    
    Pseudocode:
    Load all the npy files of normal radii and anomalous radii
    Loop through anomalous radii angles and check if the difference between
    any of them is 180.
    If it is then that must be the point of contact
    If not then it can be a defect or a bad radii due to poor resolution
    
    Check the above one by looking whether the anomalous radii angle
    is near the touching point. If not then that is simply a defect.
    
    Once all pairs are found and touching spheres are indicated
    find the resolution
    
    Input:
        nothing
    
    Output:
        Dictionary containing sphere indexes and angles indicating
        points of contact
    """
    
    # Load up the data
    radii_spheres, angles_outliers = load_data("/dls/tmp/jjl36382/analysis/")

    # touching_pts = [ ([theta1], [theta2], [phi1], [phi2]) ]
    touching_pts = {}
    
    # Loop through all pairs
    for i in range(len(radii_spheres)):
        for j in range(i + 1, len(radii_spheres)):
            # print i
            # print j
            
            ith_dict = angles_outliers[i]
            jth_dict = angles_outliers[j]
            
            touching_i_theta = []
            touching_j_theta = []
            touching_i_phi = []
            touching_j_phi = []
            
            # compare the angles
            for i_angle in ith_dict:
                for j_angle in jth_dict:
                    
                    # theta 0:360 phi 0:180
                    i_theta, i_phi = i_angle[0], i_angle[1]
                    j_theta, j_phi = j_angle[0], j_angle[1]
                    delta_theta = abs(i_theta - j_theta)
                    
                    if delta_theta == 180\
                     and approx_value(i_phi, j_phi, 180, 0):
                        touching_i_theta.append(i_theta)
                        touching_j_theta.append(j_theta)
                        touching_i_phi.append(i_phi)
                        touching_j_phi.append(j_phi)
                        
            touching_pts[i, j] = touching_i_theta, touching_j_theta,\
                                 touching_i_phi, touching_j_phi
    
    return touching_pts
    

def get_resolution(contact, indices):
    """
    Find the mean value of the angle indicating point of contact
    Loop around it within a certain range and find when radii
    become non anomalous i.e. close to the mean
    
    Input:
        Dictionary with indices of spheres and their angles of
        contact
        Indices of the spheres as a tuple
    
    Output:
        None
    """
    
    import numpy as np
    import pylab as pl
    from math import radians, cos
    # Load data
    radii_spheres, angles_outliers = load_data("/dls/tmp/jjl36382/analysis/")
    
    angles_theta1 = contact[0]
    angles_theta2 = contact[1]
    angles_phi1 = contact[2]
    angles_phi2 = contact[3]
    i1 = indices[0]
    i2 = indices[1]
    
    # Mean values of radii and angles
    mean_theta1 = np.mean(angles_theta1)
    mean_theta2 = np.mean(angles_theta2)
    mean_phi1 = np.mean(angles_phi1)
    mean_phi2 = np.mean(angles_phi2)
    mean_radii1 = np.mean(radii_spheres[i1])
    mean_radii2 = np.mean(radii_spheres[i2])
   
    """# Get the iterators for each angle range
    # They are within four stdev's of each mean
    range_theta1 = range(int(mean_theta1 - np.std(angles_theta1) * 5),\
                         int(mean_theta1 + np.std(angles_theta1) * 5))
    range_phi1 = range(int(mean_phi1 - np.std(angles_phi1) * 5),\
                       int(mean_phi1 + np.std(angles_phi1) * 5))
    range_theta2 = range(int(mean_theta2 - np.std(angles_theta2) * 5),\
                         int(mean_theta2 + np.std(angles_theta2) * 5))
    range_phi2 = range(int(mean_phi2 - np.std(angles_phi2) * 5),\
                       int(mean_phi2 + np.std(angles_phi2) * 5))
    
    # The blobs are more or less circular
    # Hence, if the middle is know from mean_theta
    # and mean_phi then only one edge is needed to
    # know the approximate resolution
    # Start at the middle and go along phi (altitude)
    # above the theta angle (horizon/azimuth)
    max_ang1 = 0
    max_ang2 = 0
    for phi in range_phi1:
        radii = radii_spheres[i1][mean_theta1, phi]
        delta = radii - mean_radii1
        if (delta > 10):
            if phi > max_ang1:
                max_ang1 = phi
                # print max_ang1
                pt1 = (radii, mean_theta1, max_ang1)
    
    # print "gap"
    
    for phi in range_phi2:
        radii = radii_spheres[i2][mean_theta2, phi]
        delta = radii - mean_radii2
        if (delta > 10):
            if phi > max_ang2:
                max_ang2 = phi
                # print max_ang2
                pt2 = (radii, mean_theta2, max_ang2)
            
    resolution = pt1[0] * (1 - cos(radians(pt1[2]))) + pt2[0] * (1 - cos(radians(pt2[2])))
    print resolution"""
    
    # -------------- use selector and plot radii techniques -------
    
    # segment the blobs out of the plot
    R1 = int(1.2 * (max(angles_theta1) - min(angles_theta1)))
    R2 = int(1.2 * (max(angles_theta2) - min(angles_theta2)))
    
    area1 = radii_spheres[i1][mean_theta1 - R1:mean_theta1 + R1,\
                              mean_phi1 - R1:mean_phi1 + R1]
    area2 = radii_spheres[i2][mean_theta2 - R2:mean_theta2 + R2,\
                              mean_phi2 - R2:mean_phi2 + R2]
    
    pl.subplot(1, 2, 1)
    pl.imshow(area1)
    from scipy.ndimage import median_filter, gaussian_filter
    from skimage.filter import sobel, threshold_otsu
    #area1 = median_filter(area1, 10)
    area1 = gaussian_filter(area1, 2)
    pl.subplot(1, 2, 2)
    pl.imshow(area1)
    pl.show()

    pl.subplot(1, 2, 1)
    pl.imshow(area2)
    pl.subplot(1, 2, 2)
    #area2 = gaussian_filter(area2, 5)
    area2 = median_filter(area2, 6)
#     area2 = sobel(area2)
#     threshold = threshold_otsu(area2)
#     area2 = area2 >= 420
    pl.imshow(area2)
    pl.show()
    
    return


contact = find_contact()
index = contact.keys()
# Take the 0th and 1st spheres
# and find the resolution between them
get_resolution(contact[0, 1], index[0])