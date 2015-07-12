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
    
    list_indices = []
    for name in radii_names:
        list_indices.append(int(name.split("radii")[1][:-4]))
    
    # translate radii numbers to file names
    index_dict = {}
    for i in range(len(radii_names)):
        index_dict[i] = list_indices[i]
        
    # a list of radii vs angle
    radii_spheres = []
    # this is a list of dicts
    angles_outliers = []
    for i in range(len(radii_names)):
        radii_spheres.append(np.load(radii_names[i]))
        
        f = open(dict_names[i], 'r')
        angles_outliers.append(pickle.load(f))
        f.close()
    
    return radii_spheres, angles_outliers, index_dict


def approx_diff(x, y, value, tolerance=5):
    """
    Check if two values are within a certain tolerance
    
    Input:
        two values and a tolerance
    
    Output:
        True boolean value
    """
    if (value - tolerance) <= abs(x - y) <= (value + tolerance):
        return True


def approx_sum(x, y, value, tolerance=5):
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
    radii_spheres, angles_outliers, index_dict = load_data("/dls/tmp/jjl36382/analysis/")

    # touching_pts = [ ([theta1], [theta2], [phi1], [phi2]) ]
    touching_pts = {}
    
    # Loop through all pairs
    for i in range(len(radii_spheres)):
        for j in range(i + 1, len(radii_spheres)):
            # print i
            # print j
            is_touching = False
            
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
                    
                    if approx_diff(i_theta, j_theta, 180, 0)\
                     and approx_sum(i_phi, j_phi, 180, 0):
                        touching_i_theta.append(i_theta)
                        touching_j_theta.append(j_theta)
                        touching_i_phi.append(i_phi)
                        touching_j_phi.append(j_phi)
                        is_touching = True
            
            if is_touching:            
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
    import os
    
    import get_blob as gb
    import blob_circles as bc
    
    # Load data
    radii_spheres, angles_outliers, index_dict = load_data("/dls/tmp/jjl36382/analysis/")
    
    angles_theta1 = contact[0]
    angles_theta2 = contact[1]
    angles_phi1 = contact[2]
    angles_phi2 = contact[3]
    i1 = indices[0]
    i2 = indices[1]
    
    
    if len(angles_theta1) < 2:
        print "not enough data obtained"
        return None
    
    # Mean values of radii and angles
    mean_theta1 = np.mean(angles_theta1)
    mean_theta2 = np.mean(angles_theta2)
    mean_phi1 = np.mean(angles_phi1)
    mean_phi2 = np.mean(angles_phi2)
    
    # This was pointless - didn't help much
    # average_theta1, average_theta2 = pair_closest_mean(angles_theta1, angles_theta2)
    # average_phi1, average_phi2 = pair_closest_mean(angles_phi1, angles_phi2)
    
    # Get the "diameters" of the blobs in both directions
    # If the blob is not circular then this method fails
    rtheta1 = max(angles_theta1) - min(angles_theta1)
    rphi1 = max(angles_phi1) - min(angles_phi1)
    rtheta2 = max(angles_theta2) - min(angles_theta2)
    rphi2 = max(angles_phi2) - min(angles_phi2)
    
    
    # Borders for the segmented blob area
    R1 = int(1.5 * max(rtheta1, rphi1))
    R2 = int(1.5 * max(rtheta2, rphi2))
    R1p = int(2.5 * rtheta1)
    R2p = int(2.5 * rtheta2)
    R1t = int(2.5 * rphi1)
    R2t = int(2.5 * rphi2)
    
    # Segment the blob areas (can be approximate)
    area1 = radii_spheres[i1][mean_theta1 - R1:mean_theta1 + R1,\
                              mean_phi1 - R1:mean_phi1 + R1]
    area2 = radii_spheres[i2][mean_theta2 - R2:mean_theta2 + R2,\
                              mean_phi2 - R2:mean_phi2 + R2]
     
#     area1 = radii_spheres[i1][mean_theta1 - R1t:mean_theta1 + R1t,\
#                               mean_phi1 - R1p:mean_phi1 + R1p]
#     area2 = radii_spheres[i2][mean_theta2 - R2t:mean_theta2 + R2t,\
#                               mean_phi2 - R2p:mean_phi2 + R2p]
#     
    # Threshold the pixels above average since
    # they are anomalous
    # Then apply a Gaussian filter to smear the
    # speckle pattern and threshold to get a
    # circular blob. It then is analysed to find
    # the mean radius
    
    # pad the edges to make the shape better for circle detection
    #area1 = np.pad(area1, 10, 'edge')
    #area2 = np.pad(area2, 10, 'edge')
    
    pl.subplot(2, 3, 1)
    pl.imshow(area1)
    
    pl.subplot(2, 3, 2)
    absolute1 = abs(area1 - np.mean(radii_spheres[i1])) + np.mean(radii_spheres[i1])
    gaus1 = gaussian_filter(absolute1, 3, mode = 'wrap')
    pl.imshow(gaus1)
    
    print np.mean(gaus1)
    print np.std(gaus1)
    
    pl.subplot(2, 3, 3)
    area1 = gaus1 >= np.mean(gaus1) + np.std(gaus1)
    pl.imshow(area1)
    
    pl.subplot(2, 3, 4)
    pl.imshow(area2)
    
    pl.subplot(2, 3, 5)
    absolute2 = abs(area2 - np.mean(radii_spheres[i2])) + np.mean(radii_spheres[i2])
    gaus2 = gaussian_filter(absolute2, 3, mode = 'wrap')
    pl.imshow(gaus2)

    pl.subplot(2, 3, 6)
    area2 = gaus2 >= np.mean(gaus2) + np.std(gaus2)
    pl.imshow(area2)
    
    # convert indices to file indices
    name1 = indices[0]
    name2 = indices[1]
    
    indices = (index_dict[name1], index_dict[name2]) 
    directory = "/dls/tmp/jjl36382/analysis/plot_{0}"
    name = "/dls/tmp/jjl36382/analysis/plot_{0}/Subplot_{1}.png"
    if not os.path.exists(directory.format(indices)):
        os.makedirs(directory.format(indices))
    pl.savefig(name.format(indices, indices))
    
    
    pl.show()
    pl.close('all')
    pl.show()
     
    # Get the centre positions of the blobs
    # using Hough transform
    C1 = bc.detect_circles(area1)
    C2 = bc.detect_circles(area2)
    
    if not C1 or not C2:
        print "Circle segmentation was unsuccessful - just use the average angles"
        radius1 = rphi1 / 2
        radius2 = rphi2 / 2
    else:
        print "centre from circle detection ", C1[0], C2[0]
        print "centre without circle detection ", (R1t, R1p), (R2t, R2p)
        C1[0] = (R1t, R1p)
        C2[0] = (R2t, R2p)
        radius1 = gb.plot_radii(area1, C1[0])
        radius2 = gb.plot_radii(area2, C2[0])
     
    print "Precise Phi angle 1 ", radius1
    print "Precise Phi angle 2 ", radius2
     
    # Distance between spheres when they become
    # barely resolved
    resolution = np.mean(radii_spheres[i1]) * (1 - cos(radians(radius1))) +\
                 np.mean(radii_spheres[i2]) * (1 - cos(radians(radius2)))
#     resolution = 1
    
    return resolution, indices

def calculate_resolutions():
    """
    Loop through all the pairs and calculate their resolution
    
    Input:
        None
    Output:
        None
    """
    
    contact = find_contact()
    key_list = contact.keys()
    print key_list
    
    for key in contact.iterkeys():
        if key != (0, 1):
            res, indices = get_resolution(contact[key], key)
            print "Spheres %s resolution is %f" % (indices, res)
            f = open('/dls/tmp/jjl36382/analysis/%i_%i.txt' % (indices[0], indices[1]), 'w')
            f.write("Resolution for spheres %i and %i is %f \n" % (indices[0], indices[1], res))
            f.close()
        
    return

calculate_resolutions()
