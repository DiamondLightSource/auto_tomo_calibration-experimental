import cPickle
import find_resolution as resolution
import optparse


if __name__ == '__main__' :
    """
    Loads up the information about the spheres
    and all the path names required.
    
    Checks if the spheres are touching and
    processes the ones that do.
    
    If their fitted centres appear to be out of
    bounds (only parts of the spheres were imaged),
    then don't use them (since there are no images to be
    loaded).
    
    
    """
    parser = optparse.OptionParser()

    (options, args) = parser.parse_args()

    # make the filename
    plots_path = args[0]
    results_paths = args[1]
    data_path = args[2]
    dim_bot = int(args[3]) - 1
    dim_top = int(args[4]) - 1
    
    f = open(results_paths + "centres.npy", 'r')
    centroids = cPickle.load(f)
    f.close()
    f = open(results_paths + "radii.npy", 'r')
    radius = cPickle.load(f)
    f.close()
    
    print """Error tolerance for the distance and radius comparison is 20.
             This can be changed inside find_contats.py"""
    touch_c, touch_pt, radii = resolution.find_contact_3D(centroids, radius, tol = 20.)
        
    print "Number of sphere pairs in contact", len(touch_c)
    
    # Loop through every pair and perform the calculations
    for i in range(len(touch_c)):
        
        c1 = touch_c[i][0]
        c2 = touch_c[i][1]
        r1 = radii[i][0]
        r2 = radii[i][1]
        
        # check if we have full sphere and not just a "bit" of it
        if (dim_top - c1[2]) < r1 or (dim_top - c2[2]) < r2 or\
           (dim_bot + c1[2]) < r1 or (dim_bot + c2[2]) < r2:
            print "centre pair", c1, c2, "are incorrectly detected"
        else:
            print "Centre pair", c1, c2, "is being processed"
            print "With radii values of ", r1, r2
            
            # for every centre pair generate a new folder
            # to store the plots
            plots_path_temp = plots_path + "/{0},{1}/".format((round(c1[0], 2), round(c1[1], 2), round(c1[2], 2)),
                                                              (round(c2[0], 2), round(c2[1], 2), round(c2[2], 2)))
            
            # Send centre positions for processing and resolution calculations
            resolution.touch_lines_3D(c1, c2, plots_path_temp, data_path, r1, r2)

