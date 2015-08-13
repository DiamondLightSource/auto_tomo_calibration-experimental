import os
from skimage import io
import numpy as np
import mhd_utils_3d as md


def save_data(filename, data):
    print("Saving data")
    f = open(filename, 'w')
    np.save(f, data)
    f.close()


if __name__ == '__main__' :

    import optparse

    parser = optparse.OptionParser()
    
    (options, args) = parser.parse_args()

    # make the filename
    output_filename = args[0]
    input_filename = args[1]
    height = int(args[2])
    
    size_img = io.imread(input_filename % 0)
    area = np.empty((size_img.shape[0], size_img.shape[1], height))
                    
    for i in range(height):
        
        input_file = input_filename % i
        print("Loading image %s" % input_file)
        
        img = io.imread(input_file)
        area[:, :, i] = img
    
    print("Saving image %s" % output_filename)
    np.save(output_filename + '.npy', area)
    
    
