import numpy as np
import pylab as pl
from scipy.fftpack import fftshift, fft, ifftshift, ifft2
from scipy.ndimage.interpolation import rotate


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


def get_projections_2D(image):
    """
    Obtain the Radon transform
    """
    
    # Project the sinogram
    sinogram = np.array([
            # Sum along one axis
            np.sum(
                # Rotate image
                rotate(image, theta, order=1, reshape=False, mode='constant', cval=0.0)
                ,axis=1) for theta in xrange(360)])
    
    pl.imshow(sinogram)
    pl.gray()
    pl.show()
    
    from PIL import Image # Import the library
    im = Image.fromarray(image) # Convert 2D array to image object
    im.save("sinogram.tif") # Save the image object as tif format
    
    # Fourier transform the rows of the sinogram
    sinogram_fft_rows = fftshift(
                                 fft(
                                    ifftshift(sinogram, axes=1
                                    )
                                 ), axes=1)
    
    pl.imshow(np.real(sinogram_fft_rows),vmin=-100,vmax=100)
    pl.gray()
    pl.show()
    
#     # Interpolate the 2D Fourier space grid from the transformed sinogram rows
#     fft2 = griddata( (srcy,srcx),
#     sinogram_fft_rows.flatten(),
#     (dsty,dstx),
#     method='cubic',
#     fill_value=0.0
#     ).reshape((S,S))
#     
#     recon = np.real(
#                     fftshift(
#                              ifft2(
#                                    ifftshift(fft2)
#                                    )
#                              )
#                     )
        
    return


def get_projections(image):
    """
    Obtain the Radon transform
    """
    xdim, ydim, zdim = image.shape
    xC = int(xdim / 2.)
    yC = int(ydim / 2.)
    
    print xC
    print yC
    
    for z in range(zdim):
        slice = image[:, :, z]
        
        # Project the sinogram
        sinogram = np.array([
                # Sum along one axis
                np.sum(
                    # Rotate image
                    rotate(slice, theta, order=1, reshape=False, mode='constant', cval=0.0)
                    ,axis=1) for theta in xrange(360)])
        
        pl.imshow(sinogram)
        pl.gray()
        pl.show()
        
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


def sphere(R1, R2, C1, C2, size):
    
    sphere = np.zeros((size, size, size))

    Xc1, Yc1, Zc1 = C1
    Xc2, Yc2, Zc2 = C2
    
    step = 1. / (size / 2.)
    Y, X, Z = np.meshgrid(np.arange(-1, 1, step), np.arange(-1, 1, step), np.arange(-1, 1, step))
    mask1 = (((X - Xc1)**2 + (Y - Yc1)**2 + (Z - Zc1)**2) < R1**2)
    mask2 = (((X - Xc2)**2 + (Y - Yc2)**2 + (Z - Zc2)**2) < R2**2)
    
    sphere[mask1] = 1
    sphere[mask2] = 1
    
    return sphere


def alpha(A, B, theta):
    return ((A**2) * (np.cos(theta))**2 + (B**2) * (np.sin(theta))**2)

    

def projection_shifted(A, B, C, theta, value, t):
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
    
  
def loop(A, B, C, value, size):
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
                proj = projection_shifted(A, B, C, angle, value, t)
                projection.append(proj)
                counter += 1

        if projection:
            sinogram[theta, :] = projection
            
    from PIL import Image # Import the library
    im = Image.fromarray(sinogram) # Convert 2D array to image object
    im.save("sinogram.tif") # Save the image object as tif format    
     
    return sinogram

# A1 = 0.3
# B1 = 0.5
# C1 = [0, 0]
# A2 = 0.5
# B2 = 0.2
# C2 = [0.2, 0.2]
# 
# scale = 255
# image = elipse2(A1, B1, C1, A2, B2, C2, scale)
# # pl.imshow(image)
# # pl.gray()
# # pl.show()
# 
# sino = loop(A1, B1, C1, 1., scale)
# # pl.imshow(sino)
# # pl.gray()
# # pl.show()
# 
# sino2 = loop(A2, B2, C2, 3., scale)
# # pl.imshow(sino2)
# # pl.gray()
# # pl.show()
# 
# print sino.shape
# print sino2.shape
# 
# angle, dim = sino.shape
# sino_plus = np.empty([360, dim])
# 
# for theta in range(360):
#     pixel1 = sino[theta, :]
#     pixel2 = sino2[theta, :]
#     
#     sino_plus[theta, :] = pixel1 + pixel2
#     
# pl.imshow(sino_plus)
# pl.gray()
# pl.show()

R1 = 0.2
R2 = 0.2
C1 = [-0.3, -0.3, 0]
C2 = [0.3, 0.3, 0]
#sphere = sphere(R1, R2, C1, C2, 255)
#get_projections(sphere)

A1 = 0.3
B1 = 0.5
C1 = [0, 0]
A2 = 0.5
B2 = 0.2
C2 = [0.2, 0.2]
 
scale = 255
image = elipse2(A1, B1, C1, A2, B2, C2, scale)
get_projections_2D(image)
    
    
    
    
    
    
    
    
    
    
    