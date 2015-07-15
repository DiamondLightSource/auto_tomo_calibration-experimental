import numpy as np
import pylab as pl
from skimage import io
import harris_edge as harris
from scipy.ndimage.filters import sobel
from skimage.filter import denoise_tv_chambolle

recon = np.load("/dls/tmp/jjl36382/complicated_data/spheres/sphere1.npy")[:,:,180]

(465, 1043, 1016)
X = 465
Y = 1043
Z = 1016
r = int(128 * 1.1)

response = harris.hessian_response(recon, 3)
 
corners = harris.get_harris_points(response, 10, 0.01)
 
harris.plot_harris_points(recon, corners)

# area = np.zeros((2*r+1, 2*r+1, 2*r+1))
# for i in range(Z - r, Z + r + 1, 4):
#     input_file = '/dls/science/groups/das/ExampleData/SphereTestData/45808/recon_%05i.tif' % i
#     img = io.imread(input_file)[X-r:X+r+1, Y-r:Y+r+1]
#     #area[:,:, i - (Z-r)] = img
#     pl.imshow(img)
#     pl.gray()
#     pl.pause(0.001)
#     print i





# N = recon.shape[2]
# 
# for slice in range(0, N, 10):
#     print slice
#     im = denoise_tv_chambolle(recon[:,:,slice], weight=0.01)
#     mag = np.hypot(sobel(im, 0), sobel(im, 1))  # magnitude
#     mag *= 255.0 / np.max(mag)  # normalize (Q&D)
#     pl.imshow(mag)
#     pl.gray()
#     pl.pause(0.001)
# 
# pl.imshow(recon[:,:,145])
# pl.gray()
# pl.show()
#          