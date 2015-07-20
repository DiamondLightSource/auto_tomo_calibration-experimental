import os
from skimage import io
import numpy as np
import mhd_utils_3d as md
import radii_angles


def save_data(filename, data):
    print("Saving data")
    f = open(filename, 'w')
    np.save(f, data)
    f.close()


if __name__ == '__main__':
    
    import optparse
    
    parser = optparse.OptionParser()
    
    parser.add_option("-x", "--xpos", dest="x_pos", help="X position of the centre of the sphere of interest", default=500, type='int')
    parser.add_option("-y", "--ypos", dest="y_pos", help="Y position of the centre of the sphere of interest", default=500, type='int')
    parser.add_option("-z", "--zpos", dest="z_pos", help="Z position of the centre of the sphere of interest", default=500, type='int')
    
    (options, args) = parser.parse_args()
    
    print "x = %i, y = %i, z = %i" % (options.x_pos, options.y_pos, options.z_pos)
    
    (options, args) = parser.parse_args()
    
    x = options.x_pos * 1.2
    y = options.y_pos * 1.2
    z = options.z_pos * 1.2
    
    # get the number of the frame to process
    task_id = int(os.environ['SGE_TASK_ID']) - 1
    
    # make the filename
    input_filename = args[0]
    output_filename = args[1] % task_id
    contact_filename = args[2] % task_id
    sigma = args[3]
    
    # load the sphere
    print("Loading image %s" % input_filename)
#     sphere = np.load(input_filename)
    sphere, meta_header = md.load_raw_data_with_mhd(input_filename)
    
    # measure radii
    radii, contact = radii_angles.plot_radii(sphere, (x, y, z), task_id, task_id + 10, sigma = 1)

    # save data
    print("Saving data %s" % output_filename)
    save_data(output_filename, radii)
    
    print("Saving data %s" % contact_filename)
    save_data(contact_filename, contact)