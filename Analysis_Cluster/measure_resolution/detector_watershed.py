import numpy as np

from scipy import ndimage as ndi
from skimage import measure
from skimage.morphology import watershed, label
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu
from skimage.restoration import denoise_tv_chambolle


def watershed_segmentation(image):
    
    image = denoise_tv_chambolle(image)
    thresh = threshold_otsu(image)
    image = (image > thresh) * 1

    distance = ndi.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),labels=image)
    markers = label(local_maxi)
    labels = watershed(-distance, markers, mask=image)
    
    centroids, radius = centres_of_mass_2D(labels)

    
    return [centroids, radius]



def centres_of_mass_2D(image):
    """
    Calculates centres of mass
    for all the labels
    """
    centroids = []
    bords = []
    areas = []
    radius = []
    
    for info in measure.regionprops(image, ['Centroid', 'BoundingBox', 'Area', 'equivalent_diameter', 'Label']): 
        
        # Skip wrong regions
        index = np.where(image==info['Label'])
        if index[0].size==0 & index[1].size==0:
            continue
        
        # TODO: change this value
        if info['Area'] > 500:
            
        
            centre = info['Centroid']
            D = info['equivalent_diameter']
            
            radius.append((D / 2.0))
            centroids.append(centre)

    return [centroids, radius]


# img = io.imread("image_02105.tif")
# a, b, c, d= watershed_segmentation(img)

