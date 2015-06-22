import pylab as pl
from skimage import io

data = io.imread('Reconstruction_0840.tif')

data = data[1200:2700,1100:2600]

pl.imshow(data, cmap=pl.cm.gray)

pl.show()

'''-------------------------------------'''

import timeit

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped
wrapped = wrapper(detect_circles, data)

timeit.timeit(wrapped, number=1)

'''-------------------------------------'''

from skimage import io
data = io.imread('recon_00160.tif')

runfile('C:\\Users\\eqp83935\\workspace\\data\\src\\real_simple_circle_detector_second_data.py')

bord, C, R = detect_circles(data)

image = data[bord[0][0]-100:bord[0][1]+100, bord[0][2]-100:bord[0][3]+100]

from skimage.filter import denoise_tv_chambolle
image = denoise_tv_chambolle(image, weight=0.008)

import numpy as np
C = np.asarray(C) + 100

'''-------------------------------------'''

runfile('C:\\Users\\eqp83935\\workspace\\data\\src\\sphere_detector.py')
import numpy as np

image = np.zeros((500,500,500))

draw_sphere(image, (100,100,100), 50, 1)
draw_sphere(image, (350,350,350), 90, 2)
draw_sphere(image, (150,260,260), 60, 0.5)

cube = image[30:170, 30:170, 30:170]

image = np.zeros((200,200,200))

draw_sphere(image, (50,50,50), 30, 1)
draw_sphere(image, (150,150,150), 30, 2)
draw_sphere(image, (70,130,130), 40, 0.5)

cube = image[10:90, 10:90, 10:90]

cube2 = np.zeros((700,700,700))
draw_sphere(cube2, (350,350,350), 300, 1)

image_saving = image.copy()

add_noise(image, 0.2)

display([(50,50,50),(150,150,150), (70,130,130)], [30,30,40])

image2 = np.zeros((1000,1000,3))
runfile('C:\\Users\\eqp83935\\workspace\\data\\src\\simple_circle_detector.py')
draw_circle(image2[:,:,1], 700, 700, 200, 0.5)
draw_circle(image2[:,:,2], 701, 705, 199, 0.5)
draw_circle(image2[:,:,0], 698, 703, 197, 0.5)
draw_circle(image2[:,:,0], 250, 250, 150, 1)
draw_circle(image2[:,:,1], 252, 245, 151, 1)
draw_circle(image2[:,:,2], 248, 249, 146, 1)

for i in range(cube2.shape[1]):
    dnp.plot.image(cube2[i,:,:])

for i in range(len(cubes[0])):
    dnp.plot.image(cubes[0][i])

'''-------------------------------------'''

cd ..\..\Documents\38644

import numpy as np
from skimage import io

data1 = io.imread('recon_00440.tif')[600:2500,100:1900]
data = np.zeros((data1.shape[0], data1.shape[1], len(range(440, 1300, 10))))
for slice in range(440, 1300, 10):
    if slice < 1000:
        data[:,:,(slice-440)/10] = io.imread('recon_00' + repr(slice) + '.tif')[600:2500,100:1900]
    else:
        data[:,:,(slice-440)/10] = io.imread('recon_0' + repr(slice) + '.tif')[600:2500,100:1900]

import cProfile
cProfile.run('cubes, centres, radii = detect_spheres(data[:,:,7:17])', 'restats')
import pstats
p = pstats.Stats('restats')
p.sort_stats('tottime')
p.print_stats()

'''-------------------------------------'''

    for alpha in np.arange(0,1+step,step):
        #points.append( image_area [int(Zc + alpha * delta_z)] [int(Xc + alpha * delta_x), int(Yc + alpha * delta_y)] )
        if np.sqrt((alpha*delta_x)**2 + (alpha*delta_y)**2 + (alpha*delta_z)**2) > rad_min:
            try:
                points.append( image_area [int(Zc + alpha * delta_z) / 10] [int(Xc + alpha * delta_x), int(Yc + alpha * delta_y)] )
            except IndexError:
                print 'Out of bound for alpha =', alpha, 'and R = ', R * alpha
                break
            print int(Xc + alpha * delta_x), int(Yc + alpha * delta_y), int(Zc + alpha * delta_z) / 10
            saved_alpha.append(alpha)
