import pylab as pl
import numpy as np
from scipy import ndimage as ndi
import find_resolution as find_res
import sort_centres as sort_cent
import mhd_utils_3d as md


from skimage import measure
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu
from scipy.ndimage.filters import median_filter
import optparse
import cPickle as pickle


def save_data(filename, data):
    print("Saving data")
    f = open(filename, 'w')
    pickle.dump(data, f)
    f.close()


def watershed_segmentation(image):

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image)
    
    return labels


def centres_of_mass_2D(image):
    """
    Calculates centres of mass
    for all the labels
    """
    centroids = []
    bords = []
    areas = []
    radius = []

    for info in measure.regionprops(image, ['Centroid', 'BoundingBox', 'equivalent_diameter']): 
        
        centre = info['Centroid']
        minr, minc, maxr, maxc = info['BoundingBox']
        D = info['equivalent_diameter']
    
        
        margin = 0
        
        radius.append((D / 2.0))
        bords.append((minr-margin, minc-margin, maxr+margin, maxc+margin))
        areas.append(image[minr-margin:maxr+margin,minc-margin:maxc+margin].copy())
        centroids.append(centre)
        
    return centroids, areas, bords, radius


def watershed_slicing(image):
    """
    Does the watershed algorithm slice by slice.
    Then use the labeled image to calculate the centres of
    mass for each slice.
    """
    image = median_filter(image, 3)
    thresh = threshold_otsu(image)
    image = (image > thresh) * 1
    
    N = len(image)
    slice_centroids = []
    slice_radius = []
    
    for i in range(N):
        
        slice = image[:, :, i]
        
        labels_slice = watershed_segmentation(slice)
        centroids, areas, bords, radius = centres_of_mass_2D(labels_slice)
        
        slice_centroids.append(centroids)
        slice_radius.append(radius)
#         if i > 49:
#             print centroids
#             pl.imshow(labels_slice)
#             pl.show()
        
    return slice_centroids, slice_radius


if __name__ == '__main__' :
    """
    Loads all of the detected parameters of segmented circles
    in a slice and appends them to a list.
    """
    parser = optparse.OptionParser()

    (options, args) = parser.parse_args()
    
    data_folder = args[0]
    tolerance = args[1]
    
    centroids = pickle.load( open( data_folder + '/centres.npy', "rb" ) )
    print centroids
    radii = pickle.load( open( data_folder + '/radii.npy', "rb" ) )
    print radii
    image, meta_header = md.load_raw_data_with_mhd(data_folder + '/image.mhd')
    print image.shape
    
    """
    Use the written functions to get the resolution
    """
    contacts = find_res.find_contact_3D(centroids, radii, tolerance)
    
    mean_widths = []
    # TODO: load image
    # TOFO: check analyse function
    for pair in contacts:
        pt1 = pair[0]
        pt2 = pair[1] 
        
        width = find_res.touch_lines_3D(pt1, pt2, image)
        mean_widths.append(width)
        
    save_data(data_folder + '/widths.txt', mean_widths)
    save_data(data_folder + '/widths.npy', mean_widths)
