import numpy as np
import pylab as pl
from math import sqrt
import os

def save_data(filename, data):
    print("Saving data")
    f = open(filename, 'w')
    np.save(f, data)
    f.close()

def save_dict(filename, data):
    import pickle
    print("Saving data")
    f = open(filename, 'w')
    pickle.dump(data, f)
    f.close()

if __name__ == '__main__' :
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("-a", "--start",
                         dest="a",
                         help="Starting angle",
                         default=500,
                         type='int')
    parser.add_option("-b", "--end",
                        dest="b",
                        help="Final angle",
                        default=500,
                        type='int')
    parser.add_option("-c", "--step",
                        dest="c",
                        help="Step size",
                        default=500,
                        type='int')


    (options, args) = parser.parse_args()

    start = options.a -1
    stop = options.b
    step = options.c
    radii_filename = args[0]
    index = int(args[1])
    
    # get the number of the frame to process
    #task_id = int(os.environ['SGE_TASK_ID'])
    
    print start
    print stop
    print step
    print radii_filename
        
    radii = []
    for i in range(start, stop, step):
        radii.append(np.load(radii_filename % i))
        
            
    radii_np = np.zeros((stop,181))
    for i in range(stop/step):
        radii_np[i*step:i*step+step,:] = radii[i]
	
	outliers = {}
	
    # Remove the anomalous radii
    radii_med = np.mean(radii_np)
    one_std_dev = np.std(radii_np)
    for i in range(start,stop):
        for j in range(0,180):
            # Values within 2stdev
            # Anomalous radii will always be bigger than the mean
            if abs(radii_np[i, j]) - abs(radii_med) > (one_std_dev*2):
            	angl = (i,j)
                outliers[angl] = radii_np[i, j]
                #radii_np[i, j] = radii_med 
                
	# save image
	output_filename = "/dls/tmp/jjl36382/analysis/outliers%02i.dat" % index
	output_angles = "/dls/tmp/jjl36382/analysis/radii%02i.npy" % index
    print("Saving data %s" % output_filename)
    print("Saving data %s" % output_angles)
    save_data(output_angles, radii_np)
    save_dict(output_filename, outliers)

    # Plot
    
    pl.imshow(radii_np.T)
    pl.title(r'Radii of real sphere as a function of 2 spherical angles $\theta$ and $\phi$',\
             fontdict={'fontsize': 16,'verticalalignment': 'bottom','horizontalalignment': 'center'})
    pl.xlabel(r'$\theta$', fontdict={'fontsize': 14,'verticalalignment': 'top','horizontalalignment': 'center'})
    pl.ylabel(r'$\phi$', fontdict={'fontsize': 14,'verticalalignment': 'bottom','horizontalalignment': 'right'}, rotation=0)
    #pl.xticks(np.arange(0, 360, 10), theta_bord)
    #pl.yticks(np.arange(0, len(phi_bord)+1, 10), phi_bord)
    pl.colorbar(shrink=0.8)
    pl.savefig("/dls/tmp/jjl36382/analysis/radii%02i_%f.png" % (index, radii_med))

    pl.show()
