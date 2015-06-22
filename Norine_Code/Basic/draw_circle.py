def draw_circle(np_image, x, y, radius, value):
    import numpy as np
    mg = np.meshgrid(np.arange(np_image.shape[0]), np.arange(np_image.shape[1]))
    mask = (((mg[0] - y)**2 + (mg[1] - x)**2) < radius**2)
    np_image[mask] = value
    np.plot.image(np_image)
    return

#Careful: the initial image has to be saved before erasing circles
def erase_circle(np_init_image, np_image, x, y, radius):
    import numpy as np
    mg = np.meshgrid(dnp.arange(np_image.shape[0]), dnp.arange(np_image.shape[1]))
    mask = (((mg[0] - y)**2 + (mg[1] - x)**2) < radius**2)
    np_image[mask] = np_init_image[mask]
    dnp.plot.image(np_image)
    return

def add_noise(np_image, amount):
    import numpy as np
    import pylab as pl
    from scipy import misc
    
    noise = np.random.randn(np_image.shape[0],np_image.shape[1])
    norm_noise = noise/np.max(noise)
    np_image = np_image + norm_noise*np.max(np_image)*amount
    pl.imshow(np_image, cmap=pl.cm.Greys)
    pl.show()
    misc.imsave("noisy_circles.png",np_image)
    return np_image
    
from PIL import Image
import numpy as np

img = np.array(Image.open("circles.png").convert('L'))

add_noise(img, 0.9)
