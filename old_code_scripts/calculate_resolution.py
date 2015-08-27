def calculate_resolution(path):
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
    import get_resolution as gr
   
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

    contact, centres = gr.find_contact(radii_spheres, angles_outliers)
    key_list = contact.keys()
    
    for key in contact.iterkeys():
        try:
            if key == (15, 21):
                print "processing spheres", key
                res, indices = gr.get_resolution(contact[key], centres[key], key, radii_spheres, index_dict, path)
                print "Spheres %s resolution is %f" % (indices, res)
                f = open(path + '/plot_{0}/Resolution_{1}_{2}.txt'.format(indices, indices[0], indices[1]), 'w')
                f.write("%f" % res)
                f.close()
        except:
            print "spheres failed", key
        
    return


if __name__ == '__main__':
    
#     import optparse
#    
#     parser = optparse.OptionParser()
#     (options, args) = parser.parse_args()
#     
#     path = args[0]
    
    path = "/dls/tmp/jjl36382/complicated_data/analysis"
    calculate_resolution(path)
    