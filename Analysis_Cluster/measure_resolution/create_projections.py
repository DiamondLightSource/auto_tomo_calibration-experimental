import numpy as np
from scipy.ndimage.interpolation import rotate
from PIL import Image # Import the library


def get_projections_3D(image, size, name, sampling):
    """
    Obtain the Radon transform
    """
    
    for i in range(size):
        # Project the sinogram
        sinogram = np.array([
                # Sum along one axis
                np.sum(
                    # Rotate image
                    rotate(image[:,:,i], theta, order=3, reshape=False, mode='constant', cval=0.0)
                    ,axis=0) for theta in np.linspace(1, 180, sampling)])
        
        print i
    
        from PIL import Image # Import the library
        recon = reconstruct(sinogram, range(angle))
        im = Image.fromarray(recon) # Convert 2D array to image object
        im.save(name % i) # Save the image object as tif format

    return sinogram



def elipse2(A1, B1, C1, value1, A2, B2, C2, value2, size):
    
    xc1, yc1 = C1
    xc2, yc2 = C2
     
    step = 1. / (size / 2.)
    Y, X = np.meshgrid(np.arange(-1, 1 + step, step), np.arange(1, -1 - step, -step))

     
    mask1 = (((X - xc1)**2 / A1**2 + (Y - yc1)**2 / B1**2) <= 1)
    mask2 = (((X - xc2)**2 / A2**2 + (Y - yc2)**2 / B2**2) <= 1)

    
    image = np.zeros((size, size))
    image[mask1] = value1
    image[mask2] = value2 
    
    
    from PIL import Image # Import the library
    im = Image.fromarray(image) # Convert 2D array to image object
    im.save("projection.tif") # Save the image object as tif format
    
    return image


def sphere(R1, R2, C1, C2, value1, value2, size):
    
    sphere = np.zeros((size, size, size))

    Xc1, Yc1, Zc1 = C1
    Xc2, Yc2, Zc2 = C2
    
    step = 1. / (size / 2.)
    X, Y, Z = np.meshgrid(np.arange(-1, 1 + step, step), np.arange(1, -1 - step, -step), np.arange(-1, 1 + step, step))
    mask1 = (((X - Xc1)**2 + (Y - Yc1)**2 + (Z - Zc1)**2) < R1**2)
    mask2 = (((X - Xc2)**2 + (Y - Yc2)**2 + (Z - Zc2)**2) < R2**2)
    
    sphere[mask1] = value1
    sphere[mask2] = value2
    
    return sphere


def alpha(A, B, theta):
    return np.sqrt((A**2) * (np.cos(theta))**2 + (B**2) * (np.sin(theta))**2)

    

def projection_shifted(A, B, C, theta, value, t):
    """
    Analytical projections for an elliptical phantom
    """
    xc, yc = C
    
    alph = alpha(A, B, theta)
    try:
        gamma = np.arctan(yc / xc)
    except:
        # arctan at infinity is 90 degrees
        gamma = np.radians(90)
        
    s = np.sqrt(xc**2 + yc**2)
    
    # first quadrant
    if yc > 0 and xc > 0:
        correction = t - s * np.cos(gamma - theta)
    # 4th quadrant
    elif xc < 0 and yc > 0:
        correction = t - s * np.cos(gamma - theta)
    else:
        correction = t + s * np.cos(gamma - theta)
    
    if abs(correction) <= alph:
        return ((2 * value * A * B) / (alph**2) * np.sqrt(alph**2 - correction**2))
    else:
        return 0


def analytical(R1, C1, value1, R2, C2, value2, R3, C3, value3, R4, C4, value4 ,size, sampling, name, z):
    """
    Get projections at every angle
    """
    ztop = max(C1[2], C2[2], C3[2], C4[2]) + max(R1, R2, R3, R4)
    zbot = min(C1[2], C2[2], C3[2], C4[2]) - max(R1, R2, R3, R4)
    angles = np.linspace(1, 180, sampling)
    indices = range(sampling)
    step = 1. / (size / 2.)
    
    z = -1.0 + step * z
    sinogram = np.zeros([sampling, size])
    
    h1 = abs(z - abs(C1[2]))
    new_r1 = np.sqrt(R1**2 - h1**2)
    h2 = abs(z - abs(C2[2]))
    new_r2 = np.sqrt(R2**2 - h2**2)
    h3 = abs(z - abs(C3[2]))
    new_r3 = np.sqrt(R3**2 - h3**2)
    h4 = abs(z - abs(C4[2]))
    new_r4 = np.sqrt(R4**2 - h4**2)
    if z >= zbot and ztop >= z:  # optimize
        for i in indices:
        
            theta = angles[i]
            angle = np.radians(theta)
            projection = []
            counter = 0
            
            for t in np.arange(-1.0, 1.0, step):
                
                if counter < size:
                    
                    proj1 = 0
                    proj2 = 0
                    proj3 = 0
                    proj4 = 0
                    
                    if R1 >= h1:
                        proj1 = projection_shifted(new_r1, new_r1, (C1[1], C1[0]), angle, value1, t)
                        
                    if R2 >= h2:
                        proj2 = projection_shifted(new_r2, new_r2, (C2[1], C2[0]), angle, value2, t)
                    if R3 >= h3:
                        proj3 = projection_shifted(new_r3, new_r3, (C3[1], C3[0]), angle, value3, t)
                    if R4 >= h4:
                        proj4 = projection_shifted(new_r4, new_r4, (C4[1], C4[0]), angle, value4, t)
        
                    proj = proj1 + proj2 + proj3 + proj4
                    projection.append(proj)
                    counter += 1
        
            sinogram[i, :] = projection
     
    return sinogram


def analytical_3D(R1, C1, value1, R2, C2, value2, size, sampling, name):
    """
    Get projections at every angle
    """
    ztop = max(C1[2], C2[2]) + max(R1, R2)
    zbot = min(C1[2], C2[2]) - max(R1, R2)
    angles = np.linspace(1, 180, sampling)
    indices = range(sampling)
    step = 1. / (size / 2.)
    
    for z in np.arange(-1.0, 1.0, step):
                
        sinogram = np.zeros([sampling, size])
        
        h1 = abs(z - abs(C1[2]))
        new_r1 = np.sqrt(R1**2 - h1**2)
        h2 = abs(z - abs(C2[2]))
        new_r2 = np.sqrt(R2**2 - h2**2)
        
        if z >= zbot and ztop >= z:  # optimize
            print z
            for i in indices:
            
                theta = angles[i]
                angle = np.radians(theta)
                projection = []
                counter = 0
                
                for t in np.arange(-1.0, 1.0, step):
                    
                    if counter < size:
                        
                        proj1 = 0
                        proj2 = 0
                        
                        if R1 >= h1:
                            proj1 = projection_shifted(new_r1, new_r1, (C1[1], C1[0]), angle, value1, t)
                            
                        if R2 >= h2:
                            proj2 = projection_shifted(new_r2, new_r2, (C2[1], C2[0]), angle, value2, t)

                        proj = proj1 + proj2
                        projection.append(proj)
                        counter += 1
        
                sinogram[i, :] = projection
            
            recon = reconstruct(sinogram, angles)
            im = Image.fromarray(recon) # Convert 2D array to image object
            im.save(name % ((z+1)/step)) # Save the image object as tif format 
        else:
            sinogram = np.zeros((size, size))
            im = Image.fromarray(sinogram) # Convert 2D array to image object
            im.save(name % ((z+1)/step)) # Save the image object as tif format
             
        print int((z+1)/step)
     
    return sinogram


def reconstruct(sinogram, angles):
    
    from skimage.transform import iradon
    sinogram = sinogram.T
    reconstruction_fbp = iradon(sinogram,theta=angles, circle=True)
    return reconstruction_fbp

