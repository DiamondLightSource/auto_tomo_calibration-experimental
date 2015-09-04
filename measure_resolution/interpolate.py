from scipy.interpolate import RegularGridInterpolator
import numpy as np
from skimage import io
import pylab as pl


data0 = io.imread("/dls/tmp/tomas_aidukas/new_recon_steel/50867/recon_noringsup/r_2015_0825_200207_images/image_00800.tif")
data1 = io.imread("/dls/tmp/tomas_aidukas/new_recon_steel/50867/recon_noringsup/r_2015_0825_200207_images/image_00801.tif")
data2 = io.imread("/dls/tmp/tomas_aidukas/new_recon_steel/50867/recon_noringsup/r_2015_0825_200207_images/image_00802.tif")

xdim = data0.shape[0]
ydim = data0.shape[1]
zdim = 3

empty_arr = np.empty((xdim, ydim, 3))

empty_arr[:, :, 0] = data0
empty_arr[:, :, 1] = data1
empty_arr[:, :, 2] = data2

x = np.linspace(0, xdim - 1, xdim)
y = np.linspace(0, ydim - 1, ydim)
z = np.linspace(0, zdim - 1, zdim)


interp = RegularGridInterpolator((x, y, z), empty_arr)

# extract the slice between the values

print interp([x, y, 1.5])