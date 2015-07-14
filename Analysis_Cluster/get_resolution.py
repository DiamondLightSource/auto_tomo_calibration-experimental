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


def find_contact(radii_spheres, angles_outliers):
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

    # touching_pts = [ ([theta1], [theta2], [phi1], [phi2]) ]
    touching_pts = {}
    centre_pts = {}
    
    # Loop through all pairs
    for i in range(len(radii_spheres) - 1):
        for j in range(i + 1, len(radii_spheres)):
            # print i
            # print j
            
            ith_dict = angles_outliers[i]
            jth_dict = angles_outliers[j]
            
            # store the touching pts
            touching_i_theta = []
            touching_j_theta = []
            touching_i_phi = []
            touching_j_phi = []
            
            # store centre pts
            centre_i_theta = []
            centre_j_theta = []
            centre_i_phi = []
            centre_j_phi = []
            
            # compare the angles
            for i_angle in ith_dict:
                for j_angle in jth_dict:
                    
                    # theta 0:360 phi 0:180
                    i_theta, i_phi = i_angle[0], i_angle[1]
                    j_theta, j_phi = j_angle[0], j_angle[1]
                    delta_theta = abs(i_theta - j_theta)
                    
                    if approx_diff(i_theta, j_theta, 180, 1)\
                     and approx_sum(i_phi, j_phi, 180, 1):
                        touching_i_theta.append(i_theta)
                        touching_j_theta.append(j_theta)
                        touching_i_phi.append(i_phi)
                        touching_j_phi.append(j_phi)
                        touching_pts[i, j] = touching_i_theta, touching_j_theta,\
                                             touching_i_phi, touching_j_phi
                    
                    if approx_diff(i_theta, j_theta, 180, 0)\
                     and approx_sum(i_phi, j_phi, 180, 0):
                        centre_i_theta.append(i_theta)
                        centre_j_theta.append(j_theta)
                        centre_i_phi.append(i_phi)
                        centre_j_phi.append(j_phi)
                        centre_pts[i, j] = centre_i_theta, centre_j_theta,\
                                            centre_i_phi, centre_j_phi
                   
    return touching_pts, centre_pts
  

def get_resolution(contact, centres, indices, radii_spheres, index_dict, path):
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
    import radii_angles as ra
    
    
    # Load data
    #radii_spheres, angles_outliers, index_dict = load_data("/dls/tmp/jjl36382/complicated_data/analysis/")
    
    centres_theta1 = centres[0]
    centres_theta2 = centres[1]
    centres_phi1 = centres[2]
    centres_phi2 = centres[3]
    
    i1 = indices[0]
    i2 = indices[1]
    
    contact_theta1 = contact[0]
    contact_theta2 = contact[1]
    contact_phi1 = contact[2]
    contact_phi2 = contact[3]
    
    # Median value of the centres
    mean_theta1 = int(np.median(centres_theta1))
    mean_theta2 = int(np.median(centres_theta2))
    mean_phi1 = int(np.median(centres_phi1))
    mean_phi2 = int(np.median(centres_phi2))
    
    # convert indices to file indices
    name1 = indices[0]
    name2 = indices[1]
    
    indices = (index_dict[name1], index_dict[name2]) 
    directory = path + "/plot_{0}"
    name = path + "/plot_{0}/Subplot_{1}.png"
    if not os.path.exists(directory.format(indices)):
        os.makedirs(directory.format(indices))

    # Segment one spot and another
 
    margin1 = 10
    starttheta1 = (mean_theta1 - margin1) / 10 * 10
    stoptheta1 = (mean_theta1 + margin1 + 10) / 10 * 10
    startphi1 = (mean_phi1 - margin1) / 10 * 10
    stopphi1 = (mean_phi1 + margin1 + 10) / 10 * 10
    
    radii1 = []
    name1 = "/dls/tmp/jjl36382/complicated_data/spheres/radii{0}/radii%03i.npy".format(index_dict[name1])
    for i in range(starttheta1, stoptheta1, 10):
        radii1.append(np.load(name1 % i))
       
    radii_np1 = np.zeros((stoptheta1 - starttheta1, stopphi1 - startphi1))
    for i in range((stoptheta1 - starttheta1) / 10):
        radii_np1[i * 10:i * 10 + 10, :] = radii1[i][:, startphi1:stopphi1]
    
    margin2 = 10
    starttheta2 = (mean_theta2 - margin2) / 10 * 10
    stoptheta2 = (mean_theta2 + margin2 + 10) / 10 * 10
    startphi2 = (mean_phi2 - margin2) / 10 * 10
    stopphi2 = (mean_phi2 + margin2 + 10) / 10 * 10
    
    radii2 = []
    name2 = "/dls/tmp/jjl36382/complicated_data/spheres/radii{0}/radii%03i.npy".format(index_dict[name2])
    for i in range(starttheta2, stoptheta2, 10):
        radii2.append(np.load(name2 % i))
       
    radii_np2 = np.zeros((stoptheta2 - starttheta2, stopphi2 - startphi2))
    for i in range((stoptheta2 - starttheta2) / 10):
        radii_np2[i * 10:i * 10 + 10, :] = radii2[i][:, startphi2:stopphi2]
        
    pl.subplot(2, 1, 1)
    pl.imshow(radii_np1.T)
    pl.subplot(2, 1, 2)
    pl.imshow(radii_np2.T)
    
    pl.show()
    
    area1 = radii_np1
    area2 = radii_np2
    
    # Smear the blobs to make them circular
    # Then threshold the Gaussian image
    
    pl.subplot(2, 3, 1)
    pl.imshow(area1.T)
    
    pl.subplot(2, 3, 2)
    pl.title("Segmented 'blobs' for Phi angle extraction")
    absolute1 = abs(area1 - np.mean(area1)) + np.mean(area1)
    gaus1 = gaussian_filter(absolute1, 3, mode = 'wrap')
    pl.imshow(gaus1.T)
    
    pl.subplot(2, 3, 3)
    area1 = gaus1 >= np.mean(gaus1) + np.std(gaus1)
    pl.imshow(area1.T)
     
    pl.subplot(2, 3, 4)
    pl.imshow(area2.T)
     
    pl.subplot(2, 3, 5)
    absolute2 = abs(area2 - np.mean(area2)) + np.mean(area2)
    gaus2 = gaussian_filter(absolute2, 3, mode = 'wrap')
    pl.imshow(gaus2.T)
# 
    pl.subplot(2, 3, 6)
    area2 = gaus2 >= np.mean(gaus2) + np.std(gaus2)
    pl.imshow(area2.T)
    

#     pl.savefig(name.format(indices, indices))
    
    pl.show()
    pl.close('all')
#     

    # Get the centre positions of the blobs
    # using Hough transform
    #C1 = bc.detect_circles(area1)
    #C2 = bc.detect_circles(area2)
 
    radius1 = gb.plot_radii(area1)
    radius2 = gb.plot_radii(area2)
    
    if radius1[0] and radius2[0]:
        print "Precise Phi angle 1 ", radius1[1]
        print "Precise Phi angle 2 ", radius2[1]
    else:
                 
        print """The resolution touch point is not
        symmetrical - hence the Phi is not uniform"""
        resolution = np.mean(radii_spheres[i1]) * (1 - cos(radians(radius1[1]))) +\
                 np.mean(radii_spheres[i2]) * (1 - cos(radians(radius2[1])))
                        
        print "Resolution from elongated angle is ", resolution
        
        resolution = np.mean(radii_spheres[i1]) * (1 - cos(radians(radius1[2]))) +\
                 np.mean(radii_spheres[i2]) * (1 - cos(radians(radius2[2])))
                 
        print "Resolution from shorter angle is ", resolution
        return resolution, indices
        

    # Distance between spheres when they become
    # barely resolved
    resolution = np.mean(radii_spheres[i1]) * (1 - cos(radians(radius1[1]))) +\
                 np.mean(radii_spheres[i2]) * (1 - cos(radians(radius2[1])))
    
    return resolution, indices

def calculate_resolutions():
    """
    Loop through all the pairs and calculate their resolution
    
    Input:
        None
    Output:
        None
    """
    import numpy as np
    import radii_angles as ra
    
    path = "/dls/tmp/jjl36382/complicated_data/analysis"

    radii_spheres, angles_outliers, index_dict = load_data("/dls/tmp/jjl36382/complicated_data/analysis/")
    
    contact, centres = find_contact(radii_spheres, angles_outliers)
    key_list = contact.keys()
    
    for key in contact.iterkeys():
        try:
            print "processing spheres", index_dict[key[0]], " ", index_dict[key[1]]
            res, indices = get_resolution(contact[key], centres[key], key, radii_spheres, index_dict, path)
            print "Spheres %s resolution is %f" % (indices, res)
            f = open(path + '/plot_{0}/Resolution_{1}_{2}.txt'.format(indices, indices[0], indices[1]), 'w')
            f.write("%f" % res)
            f.close()
        except:
            print "spheres failed", index_dict[key[0]], " ", index_dict[key[1]]
            
    return

calculate_resolutions()
