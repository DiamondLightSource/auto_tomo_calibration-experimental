from skimage import io
import detector_watershed
import numpy as np
import pylab as pl

def save_data(filename, data):
    import pickle
    print("Saving data")
    f = open(filename, 'w')
    pickle.dump(data, f)
    f.close()


def add_noise(np_image, amount):
    """
    Adds random noise to the image
    """
    noise = np.random.randn(np_image.shape[0],np_image.shape[1])
    norm_noise = noise/np.max(noise)
    np_image = np_image + norm_noise*np.max(np_image)*amount
    
    return np_image


def detect(size, name, results):
    for i in range(size):
        
        input_filename = name % i
        
        image = io.imread(input_filename)
        image = add_noise(image, 0.3)
        
        do = 0 
        if i == int(size / 2.):
            do = 1
            
        result = detector_watershed.watershed_segmentation(image, do)
        
        output_filename = results % i
        save_data(output_filename, result)
