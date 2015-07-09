import numpy as np
import pylab as pl
from skimage import io

recon = np.load("/dls/tmp/jjl36382/spheres/sphere18.npy")

"""ind = 355
for i in range(355, 500, 10):
    input_file = '/dls/science/groups/das/ExampleData/SphereTestData/45808/recon_%05i.tif' % i
    img = io.imread(input_file)
    pl.imshow(img)
    pl.gray()
    pl.pause(0.001)
    print i"""

N = recon.shape[2]


for slice in range(0, N, 10):
    print slice
    pl.imshow(recon[:,:,slice])
    pl.gray()
    pl.pause(0.001)