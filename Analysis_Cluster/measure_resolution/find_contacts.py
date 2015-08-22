import pickle
import find_resolution as resolution
import optparse

if __name__ == '__main__' :
    parser = optparse.OptionParser()

    (options, args) = parser.parse_args()

    # make the filename
    plots_path = args[0]
    results_paths = args[1]
    data_path = args[2]
    tolerance = args[3]
    dim_bot = int(args[4]) - 1
    dim_top = int(args[5]) - 1
    
    print int(tolerance)
    
    f = open(results_paths + "centres.npy", 'r')
    centroids = pickle.load(f)
    f.close()
    f = open(results_paths + "radii.npy", 'r')
    radius = pickle.load(f)
    f.close()
    
    print "centres of spheres", centroids
    print "radii of spheres", radius
    touch_c, touch_pt, radii = resolution.find_contact_3D(centroids, radius, tol = 20.)
    
    # define sampling size
    sample = 1
        
    print "spheres in contact", touch_c
    print "number of spheres in contact", len(touch_c)
    for i in range(len(touch_c)):
        c1 = touch_c[i][0]
        c2 = touch_c[i][1]
        r1 = radii[i][0]
        r2 = radii[i][1]
#         # check if we have full sphere and not just a "bit" of it
        if (dim_top - c1[2])< r1 or (dim_top - c2[2])< r2 or (dim_bot + c1[2]) < r1 or (dim_bot + c2[2]) < r2:
            print "centre pair", c1, c2, "are incorectly detected"
        else:
            print "centre pair", c1, c2, "are being processed"
            print "with radii", r1, r2
            # for every centre pair generate a new folder
            plots = plots_path + "/{0},{1}/".format(round(c1[0],2), round(c1[1],2), round(c1[2],2),
                                                    round(c2[0],2), round(c2[1],2), round(c2[2],2))
            resolution.touch_lines_3D(c1, c2, sample, plots, data_path, r1, r2)

