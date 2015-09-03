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

    (options, args) = parser.parse_args()
    
    x = options.x_pos * 1.1
    y = options.y_pos * 1.1
    z = options.z_pos * 1.1
    
    # get the number of the frame to process
    task_id = int(os.environ['SGE_TASK_ID']) - 1
    
    # make the filename
    input_filename = args[0]
    output_filename = args[1] % task_id
    contact_filename = args[2] % task_id
    width_filename = args[3] % task_id

    sigma = int(args[4])
    step = 1
    # load the sphere
    print("Loading image %s" % input_filename)
#     sphere = np.load(input_filename)
    sphere, meta_header = md.load_raw_data_with_mhd(input_filename)
    
    # measure radii
    radii, contact, widths = radii_angles.plot_radii(sphere, (x, y, z), task_id, task_id + 10, step, sigma)

    # save data
    print("Saving data %s" % output_filename)
    save_data(output_filename, radii)
    
    print("Saving data %s" % contact_filename)
    save_data(contact_filename, contact)
    
    print("Saving data %s" % contact_filename)
    save_data(width_filename, widths)
