import os
from skimage import io
import numpy as np
import create_projections as projections
from PIL import Image


if __name__ == '__main__' :

    import optparse

    parser = optparse.OptionParser()

    (options, args) = parser.parse_args()
    
    task_id = int(os.environ['SGE_TASK_ID']) - 1
    output_filename = args[0]

    R1 = 0.2
    R2 = 0.2
    C1 = (0.2, 0.2, 0.)
    C2 = (-0.2, 0.2, 0.)
    size = 2560 #total image dimensions
    sampling = 360 * 3
    
    sino = projections.analytical(R1, C1, 1., R2, C2, 1., size, sampling, output_filename, task_id)
    
    out = output_filename % task_id
    im = Image.fromarray(sino) # Convert 2D array to image object
    im.save(out) # Save the image object as tif format
    
#     angles = np.linspace(1, 180, sampling)
#     sino = io.imread(out)
#     recon = projections.reconstruct(sino, angles)
#     
#     folder = "/dls/tmp/jjl36382/resolution1/recon/recon_%05i.tif" % task_id
#     im = Image.fromarray(recon) # Convert 2D array to image object
#     im.save(folder) # Save the image object as tif format