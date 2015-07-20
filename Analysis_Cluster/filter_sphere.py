import os
import numpy as np
import mhd_utils_3d as md
import subprocess


def save_data(filename, data):
    print("Saving data")
    f = open(filename, 'w')
    np.save(f, data)
    f.close()


if __name__ == '__main__':
    
#     import optparse
#     
#     parser = optparse.OptionParser()
#     (options, args) = parser.parse_args()
#     
#     # get the number of the frame to process
#     task_id = int(os.environ['SGE_TASK_ID'])
#     
#     # make the filename
#     input_filename = args[0]
#     sphere_path = args[1]
#     sigma = args[2]
#     
#     code_path = sphere_path + "/itk_hes_rca"
#     
#     # filter the image for radii detection
#     print("Filter image %s" % input_filename)
    subprocess.call(['./itk_hes_rca', "/dls/tmp/jjl36382/complicated_data/spheres/sphere1.mhd", "4"])
    
    # save image
    #print("Saving image %s" % output_filename)
    #save_data(output_filename, sphere)
