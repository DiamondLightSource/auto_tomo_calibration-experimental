import os
from skimage import io
import numpy as np
import mhd_utils_3d as md
import create_projections as projections

def save_data(filename, data):
    print("Saving data")
    f = open(filename, 'wb')
    np.save(f, data)
    f.close()


if __name__ == '__main__' :

    import optparse

    parser = optparse.OptionParser()

    (options, args) = parser.parse_args()
    
    task_id = int(os.environ['SGE_TASK_ID']) - 1
    output_filename = args[0] % task_id

    R1=0.3
    R2=0.3
    C1=(0.3, 0.3, 0.)
    C2=(-0.3, 0.3, 0.)
    size=200 #total image dimensions
    sampling=360
    median=3
    
    projections.analytical_3D(R1, C1, 1., R2, C2, 0.8, size, sampling, output_filename)
    
