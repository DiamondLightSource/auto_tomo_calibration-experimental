import os
import pylab as pl
from skimage import io
import numpy as np


def save_data(filename, data):
    import pickle
    print("Saving data")
    f = open(filename, 'w')
    pickle.dump(data, f)
    f.close()


input_filename = "./data/analytical%i.tif"
x, y, z = (100, 100, 100)
area = np.zeros((x, y, z))

for i in range(z):
    
    input_file = input_filename % (i)    
    img = io.imread(input_file)
    area[:,:,i] = img[:,:]

print area.dtype
save_data("./data/image.npy", area)