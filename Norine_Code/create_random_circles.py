

def draw_circle_for_random(np_image, x, y, radius, value):
    
    import numpy as np
    
    mg = np.meshgrid(np.arange(np_image.shape[0]), np.arange(np_image.shape[1]))
    mask = (((mg[0] - y)**2 + (mg[1] - x)**2) < radius**2)
    np_image[mask] = value
    
    return

def random_circles(nb_circles):
    
    import numpy as np
    import random
    import pylab as pl
    from skimage.draw import circle_perimeter
    
    image = np.zeros((1000, 1000))
    
    values = []
    radii = []
    centres = []
    
    mg = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    
    for i in range(nb_circles):
        
        # Create the circle
        values.append(random.uniform(3, 10))
        radii.append(int(random.uniform(50, 200)))
        centres.append((int(random.uniform(radii[i]+10, 990-radii[i])), int(random.uniform(radii[i]+10, 990-radii[i]))))
        
        mask = (((mg[0] - centres[i][1])**2 + (mg[1] - centres[i][0])**2) < radii[i]**2)
        
        # If another circle already occupies the area, create a different circle
        while image[mask].any():
            
            radii[i] = int(random.uniform(50, 200))
            centres[i] = (int(random.uniform(radii[i]+10, 990-radii[i])), int(random.uniform(radii[i]+10, 990-radii[i])))
            
            mask = (((mg[0] - centres[i][1])**2 + (mg[1] - centres[i][0])**2) < radii[i]**2)
            
        draw_circle_for_random(image, centres[i][0], centres[i][1], radii[i]-5, values[i])
    
    pl.imshow(image, cmap=pl.cm.gray)
    pl.show()
    
    from scipy import misc
    misc.imsave("circles.png", image)
    
    parameters = [centres, radii, values]
    
    return image

random_circles(1)
