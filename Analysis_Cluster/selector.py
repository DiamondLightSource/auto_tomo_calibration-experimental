import os
from skimage import io
import numpy as np

def save_data(filename, data):
    print("Saving data")
    f = open(filename, 'w')
    np.save(f, data)
    f.close()

if __name__ == '__main__' :
    import optparse
    usage = "%prog [options] output_file_template \n" + \
        "  output_file_template = /location/of/output/filename%05i.dat"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-x", "--xpos",
                         dest="x_pos",
                         help="X position of the centre of the sphere of interest",
                         default=500,
                         type='int')
    parser.add_option("-y", "--ypos",
                        dest="y_pos",
                        help="Y position of the centre of the sphere of interest",
                        default=500,
                        type='int')
    parser.add_option("-z", "--zpos",
                        dest="z_pos",
                        help="Z position of the centre of the sphere of interest",
                        default=500,
                        type='int')
    parser.add_option("-r", "--radius",
                        dest="radius",
                        help="Radius of the sphere of interest",
                        default=200,
                        type='int')

    (options, args) = parser.parse_args()

    print "x = %i, y = %i, z = %i, r = %i" % (options.x_pos, options.y_pos, options.z_pos, options.radius)
    
    x = options.x_pos
    y = options.y_pos
    z = options.z_pos
    r = options.radius

    # get the number of the frame to process
    task_id = int(os.environ['SGE_TASK_ID'])

    # make the filename
    output_filename = args[0] % task_id
    input_filename = args[1]

    # load image
    R = int(1.2*r)
    area = np.zeros((2*R+1, 2*R+1, 2*R+1))
    for i in range(z-R, z+R+1):
        input_file = input_filename % i
        print("Loading image %s" % input_file)
        area[i-(z-R)] = io.imread(input_file)[x-R:x+R+1, y-R:y+R+1]

    # save image
    print("Saving image %s" % output_filename)
    save_data(output_filename, area)
