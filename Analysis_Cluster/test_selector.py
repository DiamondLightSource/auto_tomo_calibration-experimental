import numpy as np
import pylab as pl
from skimage import io

recon = np.load("/dls/tmp/jjl36382/complicated_data/spheres/sphere1.npy")


# 1435, 1368, 875
# (616, 470, 676)
X = 616
Y = 470
Z = 676
r = int(129 * 1.4)
# area = np.zeros((2*r+1, 2*r+1, 2*r+1))
# for i in range(Z - r, Z + r + 1):
#     input_file = '/dls/science/groups/das/ExampleData/SphereTestData/45808/recon_%05i.tif' % i
#     img = io.imread(input_file)[X-r:X+r+1, Y-r:Y+r+1]
#     area[:,:, i - (Z-r)] = img
# #     pl.imshow(img)
# #     pl.gray()
# #     pl.pause(0.001)
#     print i



N = recon.shape[2]
     
for slice in range(0, N, 10):
    print slice
    pl.imshow(recon[:,:,slice])
    pl.gray()
    pl.pause(0.001)

