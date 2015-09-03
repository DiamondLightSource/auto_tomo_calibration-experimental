import os
from skimage import io
import mhd_utils_3d as md
from scipy.ndimage.filters import median_filter
from skimage.morphology import reconstruction
from skimage.filter import threshold_otsu
import numpy as np
from skimage.morphology import watershed

from scipy import ndimage, misc
from skimage import measure
from skimage import exposure
from skimage.morphology import watershed
from skimage.filter import threshold_otsu, sobel, denoise_tv_chambolle
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter

from circle_fit import leastsq_circle


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
    results_path = args[6]
     
    # load image
    R = int(r*1.1)
    area = np.zeros((2*R + 1, 2*R + 1, 2*R + 1), dtype=np.float32)
     
    for i in range(z - R, z + R + 1):
         
        input_file = input_filename % i
        print("Loading image %s" % input_file)
         
        img = io.imread(input_file)[x - R:x + R + 1, y - R:y + R + 1]
         
        # Denoise and increase contrat of every image
        image_filtered = denoise_tv_chambolle(img, weight=0.005)
     
        float_img = rescale_intensity(image_filtered,
                                      in_range=(image_filtered.min(),
                                                image_filtered.max()),
                                      out_range='float32')
         
        p2, p98 = np.percentile(float_img, (2, 99))
        equalize = exposure.rescale_intensity(float_img,
                                              in_range=(p2, p98),
                                              out_range='float32')
         
        # Stack the normalised and de-noised spheres
        binary = (equalize > threshold_otsu(equalize)) * 1
#         filled = ndimage.morphology.binary_closing(binary, iterations=5)
        area[:, :, i - (z - R)] = binary
     
    md.write_mhd_file(output_filename + 'nothresh.mhd', area, area.shape)
    np.save(output_filename + 'nothresh.npy',area)
     
    sphere = area
    
    # Compute Euclidean Distance Map
    distance = ndimage.distance_transform_edt(sphere)
    
    # Find the maxima of EDM corresponding to the sphere centre
    max_pos = ndimage.measurements.maximum_position(distance)
    
    # Store the distance map
    md.write_mhd_file(output_filename + 'distance.mhd', distance, distance.shape)
    np.save(output_filename + 'distance.npy', distance)
    
    # Only find one highest peak from the distance map
    # i.e. the peak of the central sphere of interest
    # Labelled image has the same shape and type as markers, and since
    # markers have the same shape as the initial array then centre
    # shifts can be computed
    local_maxi = peak_local_max(distance,# num_peaks=1,
                            indices=False, labels=area)
    
    # Markers correspond to the maxima inside the EDM
    markers = ndimage.label(local_maxi)[0]
    labeled = watershed(-distance, markers, mask=area)
    
    md.write_mhd_file(output_filename + 'water.mhd', labeled, labeled.shape)
    np.save(output_filename + 'water.npy', labeled)

    label_binary = watershed(-area, markers, mask=area)
    md.write_mhd_file(output_filename + 'waterbin.mhd', labeled, labeled.shape)
    np.save(output_filename + 'waterbin.npy', labeled)
    
    # Position of the segmented sphere
    xc, yc, zc = ndimage.center_of_mass(labeled)
    
    # Initial centre estimate
    initx, inity, initz = ((2*R + 1) / 2., (2*R + 1) / 2., (2*R + 1) / 2.)
     
    # Correction of the centroids based on the initial estimate
    corrx, corry, corrz = (initx - xc, inity - yc, initz - zc)
     
    # Corrected centre values
    new_centre = (x + corrx, y + corry, z + corrz)

    radius = labeled.shape[2] / 2.
    
    f = open(results_path + '/centres_corrected.txt', 'a')
    f.write(repr(new_centre) + '\n')
    f.close()
    f = open(results_path + '/radius_corrected.txt', 'a')
    f.write(repr(radius) + '\n')
    f.close()
