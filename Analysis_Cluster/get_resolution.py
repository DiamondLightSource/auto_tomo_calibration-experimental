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
    
def pair_closest_mean(list1, list2):
    """
    Takes in two lists and finds the pair closest
    to the mean while satisfying the angle conditions:
    Difference of Theta pair must be equal to 180
    Sum of Phi pair must be equal to 180
    
    Input:
        Lists containing the angles (same length)
    Output:
        The pair closest to the mean as a tuple
    """
    
    import numpy as np
    
    for i in range(len(list1)):
        tol1 = 100
        tol2 = 100
        if (abs(list1[i] - np.mean(list1)) < tol1) and\
            (abs(list2[i] - np.mean(list2)) < tol2):
            
            tol1 = list1[i] - np.mean(list1)
            tol2 = list2[i] - np.mean(list2)
            X = list1[i]
            Y = list2[i]
    
    return (X, Y)

def get_resolution(contact, indices):
    """
    Segment the blobs at the point of contact.
    Threshold and smear them out to get a circular shape
    Use Hough transform circle detector to get their centres
    Detect the edges of these blobs and obtain their radii
    Use this radii in the resolution equation.
    This is as precise as it can get in my opinion
    using simple methods like these.
    
    Input:
        Dictionary with indices of spheres and their angles of
        contact
        Indices of the spheres as a tuple
    
    Output:
        Resolution between two touching spheres
    """
    
    import numpy as np
    import pylab as pl
    from math import radians, cos
    from scipy.ndimage import gaussian_filter
    
    import get_blob as gb
    import blob_circles as bc
    
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
    
    # This was pointless - didn't help much
    # average_theta1, average_theta2 = pair_closest_mean(angles_theta1, angles_theta2)
    # average_phi1, average_phi2 = pair_closest_mean(angles_phi1, angles_phi2)
    
    # Get the "diameters" of the blobs in both directions
    rtheta1 = max(angles_theta1) - min(angles_theta1)
    rphi1 = max(angles_phi1) - min(angles_phi1)
    rtheta2 = max(angles_theta2) - min(angles_theta2)
    rphi2 = max(angles_phi2) - min(angles_phi2)
    
    # Borders for the segmented blob area
    R1 = int(1.5 * max(rtheta1, rphi1))
    R2 = int(1.5 * max(rtheta2, rphi2))
    
    # Segment the blob areas (can be approximate)
    area1 = radii_spheres[i1][mean_theta1 - R1:mean_theta1 + R1,\
                              mean_phi1 - R1:mean_phi1 + R1]
    area2 = radii_spheres[i2][mean_theta2 - R2:mean_theta2 + R2,\
                              mean_phi2 - R2:mean_phi2 + R2]
    
    # Threshold the pixels above average since
    # they are anomalous
    # Then apply a Gaussian filter to smear the
    # speckle pattern and threshold to get a
    # circular blob. It then is analysed to find
    # the mean radius
    pl.subplot(2, 2, 1)
    pl.imshow(area1)
        
    pl.subplot(2, 2, 2)
    absolute1 = abs(area1 - np.mean(radii_spheres[i1])) + np.mean(radii_spheres[i1])
    area1 = gaussian_filter(absolute1, 2)
    area1 = area1 >= np.mean(area1)
    pl.imshow(area1)
    
    pl.subplot(2, 2, 3)
    pl.imshow(area2)
    
    pl.subplot(2, 2, 4)
    absolute2 = abs(area2 - np.mean(radii_spheres[i2])) + np.mean(radii_spheres[i2])
    area2 = gaussian_filter(absolute2, 2)
    area2 = area2 >= np.mean(area2)
    pl.imshow(area2)
    
    pl.show()
    
    # Get the centre positions of the blobs
    # using Hough transform
    C1 = bc.detect_circles(area1)
    C2 = bc.detect_circles(area2)
    print "centre from circle detection ", C1[0], C2[0]
    
    radius1 = gb.remove_large_sine(area1, C1[0])
    radius2 = gb.remove_large_sine(area2, C2[0])
    
    print "Radius1 using min/max ", rtheta1 / 2.0
    print "Radius2 using min/max ", rtheta2 / 2.0
    print "Precise Radius1 ", radius1
    print "Precise Radius2 ", radius2
    
    # Distance between spheres when they become
    # barely resolved
    resolution = np.mean(radii_spheres[i1]) * (1 - cos(radians(radius1))) +\
                 np.mean(radii_spheres[i2]) * (1 - cos(radians(radius2)))
    
    return resolution

def calculate_resolutions():
    """
    Loop through all the pairs and calculate their resolution
    
    Input:
        None
    Output:
        None
    """
    
    contact = find_contact()
    index = contact.keys()
    
    for i in range(len(index)):
        print get_resolution(contact[index[i]], index[i])
    
    return

calculate_resolutions()