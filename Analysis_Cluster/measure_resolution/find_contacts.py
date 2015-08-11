import pickle
import find_resolution as resolution

if __name__ == '__main__' :
    
    f = open(sorted + "centres.npy", 'r')
    centroids = pickle.load(f)
    f.close()
    f = open(sorted +"radii.npy", 'r')
    radius = pickle.load(f)
    f.close()
    
    print "centres of spheres", centroids
    print "radii of spheres", radius
    touch_c, touch_pt, radii = resolution.find_contact_3D(centroids, radius, tol = 5.)
    
    # define sampling size
    sample = 2
        
    for i in range(len(touch_c)):
        c1 = touch_c[i][0]
        c2 = touch_c[i][1]
        r1 = radii[i][0]
        r2 = radii[i][1]
        
        plots = "/dls/tmp/jjl36382/resolution1/plots" 
        name = ???????????????
        resolution.touch_lines_3D(c1, c2, sample, plots, name)

