import numpy as np
import pylab as pl
from skimage import io

#recon = np.load("/dls/tmp/jjl36382/spheres/sphere2.npy")


# 1435, 1368, 875
# for i in range(500, 1200, 20):
#     input_file = '/dls/science/groups/das/ExampleData/SphereTestData/38644/recon_%05i.tif' % i
#     img = io.imread(input_file)
#     pl.imshow(img)
#     pl.gray()
#     pl.pause(0.001)
#     print i

# i = 825
# input_file = '/dls/science/groups/das/ExampleData/SphereTestData/38644/recon_%05i.tif' % i
# img = io.imread(input_file)
# pl.imshow(img)
# pl.gray()
# pl.show()
    
        
    
"""

N = recon.shape[2]


for slice in range(0, N, 10):
    print slice
    pl.imshow(recon[:,:,slice])
    pl.gray()
    pl.pause(0.001)"""