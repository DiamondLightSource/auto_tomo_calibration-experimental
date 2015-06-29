import os
from skimage import io
import numpy as np
from scipy.ndimage import median_filter,gaussian_filter
from skimage.restoration import denoise_tv_chambolle

def save_data(filename, data):
    print("Saving data")
    f = open(filename, 'w')
    np.save(f, data)
    f.close()

if __name__ == '__main__' :
    import optparse
    usage = "%prog [options] input_file_template, output_file_template \n" + \
        "  input_file_template  = /location/of/file/filename%02i.npy \n" + \
        "  output_file_template = /location/of/output/filename%02i.dat"
    parser = optparse.OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    print 'test'
    # get the number of the frame to process
    task_id = int(os.environ['SGE_TASK_ID'])
    
    # make the filename
    input_filename = args[0] % task_id
    output_filename = args[1] % task_id
    
    # load the sphere
    print("Loading image %s" % input_filename)
    sphere = np.load(input_filename)
    
    # filter the image for radii detection
    print("Filter image %s" % input_filename)
    #sphere = median_filter(sphere, 10)
    #sphere = gaussian_filter(sphere,3)
    sphere = gaussian_filter(sphere, 3)
    
    # save image
    print("Saving image %s" % output_filename)
    save_data(output_filename, sphere)
