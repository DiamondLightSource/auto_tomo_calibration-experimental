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


def approx_phi(x, y, tolerance=5):
    """
    Check if the phi angle is within a certain tolerance
    
    Input:
        two angles and a tolerance
    
    Output:
        True boolean value
    """
    if (180 - tolerance) <= (x + y) <= (180 + tolerance):
        return True


def get_resolution():
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
    
    Once the point of contact is known, then the function will go along one direction
    and check at what angle the radii become normal. Using this angle for
    both spheres which are touching it is possible to calculate the
    distance where they can be resolved. This distance is the resolution.
    
    Pseudocode:
    Load all the npy files of normal radii and anomalous radii
    Loop through anomalous radii angles and check if the difference between
    any of them is 180.
    If it is then that must be the point of contact
    If not then it can be a defect or a bad radii due to poor resolution
    
    Check the above one by looking whether the anomalous radii angle
    is near the touching point. If not then that is simply a defect.
    
    Once all pairs are found and touching spheres are indicated find the resolution. 
    
    Input:
        nothing
    
    Output:
        None
    """
    
    import os
    import numpy as np
    import pylab as pl
    import pickle

    
    # Load up the data    
    radii_spheres, angles_outliers = load_data("/dls/tmp/jjl36382/analysis/")
    
    
        
    # Find the touching angle position
    # touching_pts = [ ([theta1], [theta2][phi1], [phi2]) ]
    touching_pts = {}
    
    # Loop through all pairs
    for i in range(len(radii_spheres)):
        for j in range(i + 1, len(radii_spheres)):
            print i
            print j
            
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
                    delta_phi = i_phi + j_phi
                    
                    if delta_theta == 180 and approx_phi(i_phi, j_phi, 0):
                        touching_i_theta.append(i_theta)
                        touching_j_theta.append(j_theta)
                        touching_i_phi.append(i_phi)
                        touching_j_phi.append(j_phi)
                        
            touching_pts[i, j] = touching_i_theta, touching_j_theta, touching_i_phi, touching_j_phi
            
    print touching_pts
    
    print np.mean(radii_spheres[0])
    print np.mean(radii_spheres[1])
    
    return touching_pts
    
    
get_resolution()