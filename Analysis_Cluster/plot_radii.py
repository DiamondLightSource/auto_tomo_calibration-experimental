import numpy as np
import pylab as pl
from math import sqrt
import os
from scipy.ndimage import gaussian_filter


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

    start = options.a - 1
    stop = options.b
    step = options.c
    radii_filename = args[0]
    index = int(args[1])
    anal_path = args[2]
    
    # get the number of the frame to process
    #task_id = int(os.environ['SGE_TASK_ID'])
    
    print start
    print stop
    print step
    print radii_filename
        
    radii = []
    for i in range(start, stop, step):
        radii.append(np.load(radii_filename % i))

    radii_np = np.zeros((stop,180))
    for i in range(stop/step):
        radii_np[i*step:i*step+step,:] = radii[i]
	
    # Dictionary to store the angles and outlier values
	outliers = {}
	
    # Remove the anomalous radii
    radii_mean = np.mean(radii_np)
    one_std_dev = np.std(radii_np)
    
    # Get all radii above the mean
    absolute1 = abs(radii_np - radii_mean) + radii_mean
    
    # Apply a Gaussian filter
    gaus1 = gaussian_filter(absolute1, 3)
    
    # Threshold the image
    area1 = gaus1 >= np.mean(gaus1) + np.std(gaus1) * 3
    area1 = area1 * 1
    
    # Store the angles of the anomalous values
    for i in range(start,stop):
        for j in range(0,180):
            if area1[i, j] == 1:
            	angl = (i,j)
                outliers[angl] = area1[i, j] 
                
	# save image
	output_filename = anal_path + "/outliers%02i.dat" % index
	output_angles = anal_path + "/radii%02i.npy" % index
    print("Saving data %s" % output_filename)
    print("Saving data %s" % output_angles)
    save_data(output_angles, radii_np)
    save_dict(output_filename, outliers)

    # Plot
    pl.subplot(2, 1, 1)
    pl.imshow(radii_np.T)
    
    pl.title(r'Radii of real sphere as a function of 2 spherical angles $\theta$ and $\phi$',\
             fontdict={'fontsize': 16,'verticalalignment': 'bottom','horizontalalignment': 'center'})
    pl.xlabel(r'$\theta$', fontdict={'fontsize': 14,'verticalalignment': 'top','horizontalalignment': 'center'})
    pl.ylabel(r'$\phi$', fontdict={'fontsize': 14,'verticalalignment': 'bottom','horizontalalignment': 'right'}, rotation=0)
    #pl.xticks(np.arange(0, 360, 10), theta_bord)
    #pl.yticks(np.arange(0, len(phi_bord)+1, 10), phi_bord)
    pl.colorbar(shrink=0.8)
    
    pl.subplot(2, 1, 2)
    pl.imshow(area1.T)
    pl.xlabel(r'$\theta$', fontdict={'fontsize': 14,'verticalalignment': 'top','horizontalalignment': 'center'})
    pl.ylabel(r'$\phi$', fontdict={'fontsize': 14,'verticalalignment': 'bottom','horizontalalignment': 'right'}, rotation=0)

    pl.colorbar(shrink=0.8)

    pl.savefig(anal_path + "/radii%02i_%f.png" % (index, radii_mean))

    pl.show()
