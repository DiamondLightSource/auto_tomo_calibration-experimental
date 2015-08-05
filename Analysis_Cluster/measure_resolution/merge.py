import os
from skimage import io
import numpy as np
import mhd_utils_3d as md


def save_data(filename, data):
    print("Saving data")
    f = open(filename, 'wb')
    np.save(f, data)
    f.close()


if __name__ == '__main__' :

    import optparse

    parser = optparse.OptionParser()

    (options, args) = parser.parse_args()

    output_filename = args[0]
    input_filename = args[1]
    start = int(args[2]) - 1
    end = int(args[3]) - 1
    
    # Glue all tifs into one large npy file
    z = end - start
    
    input_file = input_filename % start
    print("Loading image %s" % input_file)
    
    img = io.imread(input_file)
    
    x, y = img.shape[0], img.shape[1]
    
    area = np.zeros((x, y, z))
    
    for i in range(0, z):
        
        input_file = input_filename % (i + start)
        print("Loading image %s" % input_file)
        
        img = io.imread(input_file)
        area[:,:,i] = img[:,:]
    
    print "trying to save data"
    md.write_mhd_file(output_filename + '/image.mhd', area, area.shape)
