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
    contact_filename = args[3]
    
    # get the radius  
    radii = []
    for i in range(start, stop, step):
        radii.append(np.load(radii_filename % i))
        
    radii_np = np.zeros((stop,180))
    for i in range(stop/step):
        radii_np[i*step:i*step+step,:] = radii[i]
            
    
    # get the contact points
    contact = []
    for i in range(start, stop, step):
        contact.append(np.load(contact_filename % i))

    contact_np = np.zeros((stop,180))
    for i in range(stop/step):
        contact_np[i*step:i*step+step,:] = contact[i]

    radius = np.mean(radii_np)
#     # Dictionary to store the angles and outlier values
#     outliers = {}
# 
#     # Threshold the image
#     area1 = radii_np == 0
#     area1 = area1 * 1
# 
#     # Store the angles of the anomalous values
#     for i in range(start,stop):
#         for j in range(0,180):
#             if radii_np[i, j] == 0:
#                 angl = (i,j)
#                 outliers[angl] = 0
#                 
#     # save image
#     output_filename = anal_path + "/outliers%02i.dat" % index
#     output_angles = anal_path + "/radii%02i.npy" % index
#     print("Saving data %s" % output_filename)
#     print("Saving data %s" % output_angles)
#     save_data(output_angles, radii_np)
#     save_dict(output_filename, outliers)

    # just contact pts
#     delete_list = []
#     for i in range(start,stop):
#         for j in range(0,180):
#             if contact_np[i, j] == 0:
#                 delete_list.append((i, j))
#     

    # Plot
    pl.subplot(2, 1, 1)
    pl.imshow(radii_np.T)
    
    pl.title(r'Radii of real sphere as a function of 2 spherical angles $\theta$ and $\phi$',\
             fontdict={'fontsize': 16,'verticalalignment': 'bottom','horizontalalignment': 'center'})
    pl.xlabel(r'$\theta$', fontdict={'fontsize': 14,'verticalalignment': 'top','horizontalalignment': 'center'})
    pl.ylabel(r'$\phi$', fontdict={'fontsize': 14,'verticalalignment': 'bottom','horizontalalignment': 'right'}, rotation=0)
    pl.colorbar(shrink=0.8)
    
    pl.subplot(2, 1, 2)
    pl.imshow(contact_np.T)
    pl.xlabel(r'$\theta$', fontdict={'fontsize': 14,'verticalalignment': 'top','horizontalalignment': 'center'})
    pl.ylabel(r'$\phi$', fontdict={'fontsize': 14,'verticalalignment': 'bottom','horizontalalignment': 'right'}, rotation=0)

    pl.colorbar(shrink=0.8)

    pl.savefig(anal_path + "/radii%02i_%f.png" % (index, radius))

    pl.show()
