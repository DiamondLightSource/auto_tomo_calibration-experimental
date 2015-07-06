import os
from skimage import io
import numpy as np
import h5py
import pylab as pl

import circle_detector


def save_data(filename, data):
    import pickle
    print("Saving data")
    f = open(filename, 'w')
    pickle.dump(data, f)
    f.close()


if __name__ == '__main__':
    
    import optparse
    parser = optparse.OptionParser()

    (options, args) = parser.parse_args()

    # get the number of the frame to process
    task_id = int(os.environ['SGE_TASK_ID']) - 1

    # make the filename
    input_filename = args[0]
    output_filename = args[1] % task_id
    
    f = h5py.File(input_filename, 'r')
    PATH = '/entry/instrument/detector/data/'
    s = f[PATH]
    #nb_slice = s.shape[0] - 1
    
    # load image
    #print("Loading image %s" % input_filename)
    #image = io.imread(input_filename)
    image = s[task_id]
    
    # process image
    print("Processing data")
    #time.sleep(10)
    result = circle_detector.detect_circles(image)
#    result = ""

    # save image
    print("Saving image %s" % output_filename)
    save_data(output_filename, result)
