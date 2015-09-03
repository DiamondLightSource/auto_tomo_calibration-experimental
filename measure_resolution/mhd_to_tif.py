import numpy as np
from skimage.draw import circle_perimeter
from skimage.transform import hough_circle
import pylab as pl
from scipy import ndimage, misc
import mhd_utils_3d as md

from skimage import io
from skimage import measure
from skimage import exposure
from scipy.ndimage.filters import median_filter
from skimage.morphology import watershed
from skimage.filter import threshold_otsu, sobel, denoise_tv_chambolle
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
from sphere_fit import leastsq_sphere


def save_data(filename, data):
    print("Saving data")
    f = open(filename, 'w')
    np.save(f, data)
    f.close()
    
if __name__ == '__main__' :
 
    import optparse
    import os
     
    parser = optparse.OptionParser()
     
 
    (options, args) = parser.parse_args()
 
    # get the number of the frame to process
    task_id = int(os.environ['SGE_TASK_ID'])
 
    # make the filename
    folder_name = args[0]
    output_folder = args[1]

#     sphere, meta_header = md.load_raw_data_with_mhd(folder_name + "/gradientgauss.mhd")
    
    sphere, header = md.load_raw_data_with_mhd(folder_name + "/sphere1.mhd")
    sphere = median_filter(sphere, 3)
    np_image = (sphere > threshold_otsu(sphere)) * 1
    
    import numpy as np
    from skimage.morphology import reconstruction
    
    seed = np.copy(np_image)
    seed[1:-1, 1:-1] = np_image.max()
    mask = np_image
    
    np_image = reconstruction(seed, mask, method='erosion')
    
    from scipy import ndimage
    distance = ndimage.distance_transform_edt(np_image)
    max_pos = ndimage.measurements.maximum_position(distance)
     
    print max_pos

    for z in range(np_image.shape[0]):
         
        slice = distance[z, :, :]
         
        misc.imsave(folder_name + '/slice{0}.tif'.format(z), slice)