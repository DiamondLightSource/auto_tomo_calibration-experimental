import os
from skimage import io
from scipy.ndimage.filters import median_filter
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

    # get the number of the frame to process
    task_id = int(os.environ['SGE_TASK_ID'])

    # make the filename
    x = int(float(args[0]))
    y = int(float(args[1]))
    z = int(float(args[2]))
    r = int(float(args[3]))
    output_filename = args[4]
    input_filename = args[5]
    
    # load image
    R = int(r * 1.1)
    area = np.zeros((2*R + 1, 2*R + 1, 2*R + 1))
    
    for i in range(z - R, z + R + 1):
        
        input_file = input_filename % i
        print("Loading image %s" % input_file)
        
        img = io.imread(input_file)[x - R:x + R + 1, y - R:y + R + 1]
        area[:, :, i - (z - R)] = img
    
    np.save(output_filename + '.npy', area)
    
    new_area = median_filter(area, 3)
    print("Saving image %s" % output_filename)
    md.write_mhd_file(output_filename + '.mhd', new_area, new_area.shape)
    
