import numpy as np
import pylab as pl
from scipy import fft, ifft



def add_noise(np_image, amount):
    """
    Adds random noise to the image
    """
    noise = np.random.randn(np_image.shape[0],np_image.shape[1],np_image.shape[2])
    norm_noise = noise/np.max(noise)
    np_image = np_image + norm_noise*np.max(np_image)*amount
    
    return np_image


def line_eqn(x, theta, t, xC, yC):
    """
    Why division by sine is not allowed?
    """
    if theta != 0:
        y = (t  - x * np.cos(theta))# / np.sin(theta)
        return y
    else:
        return t
        

def get_projections(image):
    """
    Obtain the Radon transform
    """
    xdim, ydim, zdim = image.shape
    xC = int(xdim / 2.)
    yC = int(ydim / 2.)
    
    print xC
    print yC
    
    diag = int(np.sqrt(xdim**2 + ydim**2) / 2.)
    diag = np.min((xC, yC))
    print diag

    for z in range(50, 51):
        slice = image[:, :, z]
        projections = []
        
        for theta in range(0, 180):
            radon = []

            angle = np.radians(theta)
            for t in range(-diag, diag):
                line_t = []
                
                for x in range(-diag, diag):
                    
                    try:
                        y = line_eqn(x, angle, t, xC, yC) + yC
                        x = x + xC
                        # store pixel values along a line at t
                        pixel = slice[x, y]
                        line_t.append(pixel)
                    except:
                        continue
                        
#                     pl.plot(line_t)
#                     pl.show()
                    # store all lines along one angle
                
                sum = np.sum(line_t)
                if sum != 0:
                    radon.append(sum)

            # store lines obtained over all angles
            # at one slice
            projections.append(np.fft.fftshift(np.fft.fft(radon)))
            
            pl.plot(np.fft.fftshift(np.fft.fft(radon)))
            #pl.ylim(-100, 100)
            pl.gray()
            pl.show()
            # TODO: store the lines at one slice over all slices
    return
 

def elipse(A, B, size):
    step = 1. / (size / 2.)
    Y, X = np.meshgrid(np.arange(-1, 1, step), np.arange(-1, 1, step))
    mask = (((X)**2 / A**2 + (Y)**2 / B**2) <= 1)
    
    image = np.zeros((size, size))
    image[mask] = 1
    
    from PIL import Image # Import the library
    im = Image.fromarray(image) # Convert 2D array to image object
    im.save("projection.tif") # Save the image object as tif format
    
    return image


def elipse2(A1, B1, C1, A2, B2, C2, size):
    
    xc1, yc1 = C1
    xc2, yc2 = C2
    
    step = 1. / (size / 2.)
    Y, X = np.meshgrid(np.arange(-1, 1, step), np.arange(-1, 1, step))
    mask1 = (((X - xc1)**2 / A1**2 + (Y - yc1)**2 / B1**2) <= 1)
    mask2 = (((X - xc2)**2 / A2**2 + (Y - yc2)**2 / B2**2) <= 1)
    
    image = np.zeros((size, size))
    image[mask1] = 1.
    image[mask2] = 3.

    from PIL import Image # Import the library
    im = Image.fromarray(image) # Convert 2D array to image object
    im.save("projection.tif") # Save the image object as tif format
    
    return image


def sphere(R1, R2, size, centre1, centre2):
    
    sphere = np.zeros((size, size, size))

    Xc1 = centre1[0]
    Yc1 = centre1[1]
    Zc1 = centre1[2]
    
    Xc2 = centre2[0]
    Yc2 = centre2[1]
    Zc2 = centre2[2]
    
    step = 1. / (size / 2.)
    Y, X, Z = np.meshgrid(np.arange(-1, 1, step), np.arange(-1, 1, step), np.arange(-1, 1, step))
    mask1 = (((X - Xc1)**2 + (Y - Yc1)**2 + (Z - Zc1)**2) < R1**2)
    mask2 = (((X - Xc2)**2 + (Y - Yc2)**2 + (Z - Zc2)**2) < R2**2)
    
    sphere[mask1] = 1
    sphere[mask2] = 2
    
#     from PIL import Image # Import the library
#     im = Image.fromarray(image) # Convert 2D array to image object
#     im.save("projection.tif") # Save the image object as tif format
    
    return sphere


def alpha(A, B, theta):
    return ((A**2) * (np.cos(theta))**2 + (B**2) * (np.sin(theta))**2)
       
       
def projection_elipse(A, B, theta, value, t):
    """
    Analytical projections for an elliptical phantom
    """
    alph = alpha(A, B, theta)
    
    if abs(t) <= alph:
        return (2 * value * A * B / (alph**2) * np.sqrt(alph**2 - t**2))
    else:
        return 0
    

def projection_shifted(A, B, C, theta, value, t, rotated):
    """
    Analytical projections for an elliptical phantom
    """
    xc, yc = C
    
    alph = alpha(A, B, theta)
    try:
        gamma = np.arctan(np.radians(yc / xc))
    except:
        # arctan at infinity is 90 degrees
        gamma = np.radians(90)
        
    s = np.sqrt(xc**2 + yc**2)
    
    correction = t - s * np.cos(gamma - theta)
    
    if abs(correction) <= alph:
        return (2 * value * A * B / (alph**2) * np.sqrt(alph**2 - correction**2))
    else:
        return 0
    
  
def loop(A, B, C, value, size, rotated):
    """
    Get projections at every angle
    """
    sinogram = np.empty([360, size])

    for theta in range(360):
    
        angle = np.radians(theta)
        a = alpha(A, B, angle)
        step = a / (size / 2.)
        
        projection = []
        counter = 0

        for t in np.arange(-a, a, step):
        #for t in np.arange(-1., 1., 0.5 / size):
            
            if counter < size:
                proj = projection_shifted(A, B, C, angle, value, t, rotated)
                projection.append(proj)
                counter += 1

        if projection:
            sinogram[theta, :] = projection
            
    from PIL import Image # Import the library
    im = Image.fromarray(sinogram) # Convert 2D array to image object
    im.save("sinogram.tif") # Save the image object as tif format    
     
    return sinogram

A1 = 0.3
B1 = 0.5
C1 = [0, 0]
A2 = 0.5
B2 = 0.2
C2 = [0.2, 0.2]

scale = 255
image = elipse2(A1, B1, C1, A2, B2, C2, scale)
# pl.imshow(image)
# pl.gray()
# pl.show()

sino = loop(A1, B1, C1, 1., scale, 1)
# pl.imshow(sino)
# pl.gray()
# pl.show()

sino2 = loop(A2, B2, C2, 3., scale, 1)
# pl.imshow(sino2)
# pl.gray()
# pl.show()

print sino.shape
print sino2.shape

angle, dim = sino.shape
sino_plus = np.empty([360, dim])

for theta in range(360):
    pixel1 = sino[theta, :]
    pixel2 = sino2[theta, :]
    
    sino_plus[theta, :] = pixel1 + pixel2
    
pl.imshow(sino_plus)
pl.gray()
pl.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    