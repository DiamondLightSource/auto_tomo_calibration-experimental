def get_resolution():
    """
    Loads up the dictionary containing the outlier radii
    and their angles. This will load up all the outlier arrays
    for all of the spheres.
    
    If the radii are known of the spheres and they are touching
    then we need to calculate the touching point using the outlier data.
    
    The angle difference must equate to 180 degrees (in 1D). This then means
    that at that point the spheres are touching each other.
    
    If anomalous radii do not have 180 degree angle then they might be defects
    of the sphere
    
    Once the point of contact is known, then the function will go along one direction
    and check at what angle the radii become normal. Using this angle for
    both spheres which are touching it is possible to calculate the
    distance where they can be resolved. This distance is the resolution.
    
    Pseudoode:
    Load all the npy files of normal radii and anomalous radii
    Loop through anomalous radii angles and check if the difference between
    any of them is 180.
    If it is then that must be the point of contact
    If not then it can be a defect or a bad radii due to poor resolution
    
    Check the above one by looking whether the anomalous radii angle
    is near the touching point. If not then that is simply a defect.
    
    Once all pairs are found and touching spheres are indicated find the resolution. 
    """
    
    import os
    
    # Load up the data    
    data_list = get_data_list("/dls/tmp/jjl36382/analysis/")
    
    for file in data_list:
        print file
#     output_filename = "/dls/tmp/jjl36382/analysis/outliers%02i.npy" % task_id
#     output_angles = "/dls/tmp/jjl36382/analysis/radii%02i.npy" % task_id
#     save_data(output_angles, radii_np)
#     save_data(output_filename, outliers)
    return
    
def get_data_list(path):
    """
    Gets the file names inside the directory
    """
    import os
    
    storage = []
    for f in os.listdir(path):
        if f.endswith(".npy"):
            storage.append(os.path.join(path,f)) 
            
    return storage

get_resolution()