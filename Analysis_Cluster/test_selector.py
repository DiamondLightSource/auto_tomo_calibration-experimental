import numpy as np
import pylab as pl
from skimage import io

recon = np.load("/dls/tmp/jjl36382/complicated_data/spheres/sphere25.npy")


# 1435, 1368, 875
# (616, 470, 676)
(465, 1043, 976)
(465, 1043, 1016)
X = 465
Y = 1043
Z = 1016
r = int(128 * 1.1)

#Z shoudl be about 1020 for sphere 1

# area = np.zeros((2*r+1, 2*r+1, 2*r+1))
# for i in range(Z - r, Z + r + 1, 4):
#     input_file = '/dls/science/groups/das/ExampleData/SphereTestData/45808/recon_%05i.tif' % i
#     img = io.imread(input_file)[X-r:X+r+1, Y-r:Y+r+1]
#     #area[:,:, i - (Z-r)] = img
#     pl.imshow(img)
#     pl.gray()
#     pl.pause(0.001)
#     print i



N = recon.shape[2]
# print N
# for slice in range(0, 50, 1):
#     print slice
#     pl.imshow(recon[:,:,slice])
#     pl.gray()
#     pl.pause(0.001)
     
for slice in range(0, N, 10):
    print slice
    pl.imshow(recon[:,:,slice])
    pl.gray()
    pl.pause(0.001)

