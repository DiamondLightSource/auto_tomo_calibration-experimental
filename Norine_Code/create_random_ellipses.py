def draw_ellipse_for_random(np_image, Xc, Yc, a, b, value):
    
    import numpy as np
    
    mg = np.meshgrid(dnp.arange(np_image.shape[0]), dnp.arange(np_image.shape[1]))
    mask = ((((mg[0] - Yc)*a)**2 + ((mg[1] - Xc)*b)**2) < (a*b)**2)
    np_image[mask] = value
    
    return

def random_ellipses(nb_ellipses):
    
    import numpy as np
    import random
    from skimage.draw import circle_perimeter
    
    image = np.zeros((1000, 1000))
    
    values = []
    a = []
    b= []
    centres = []
    
    mg = np.meshgrid(dnp.arange(image.shape[0]), dnp.arange(image.shape[1]))
    
    for i in range(nb_ellipses):
        
        # Create the circle
        values.append(random.uniform(3, 10))
        a.append(int(random.uniform(50, 200)))
        b.append(int(random.uniform(50, 200)))
        centres.append((int(random.uniform(max(a[i], b[i])+10, 990-max(a[i], b[i]))), int(random.uniform(max(a[i], b[i])+10, 990-max(a[i], b[i])))))
        
        mask = ((((mg[0] - centres[i][1])*a[i])**2 + ((mg[1] - centres[i][0])*b[i])**2) < (a[i]*b[i])**2)
        
        # If another circle already occupies the area, create a different circle
        while image[mask].any():
            
            a.append(int(random.uniform(50, 200)))
            b.append(int(random.uniform(50, 200)))
            centres[i] = (int(random.uniform(max(a[i], b[i])+10, 990-max(a[i], b[i]))), int(random.uniform(max(a[i], b[i])+10, 990-max(a[i], b[i]))))
            
            mask = ((((mg[0] - centres[i][1])*a[i])**2 + ((mg[1] - centres[i][0])*b[i])**2) < (a[i]*b[i])**2)
            
        draw_ellipse_for_random(image, centres[i][0], centres[i][1], a[i]-5, b[i]-5, values[i])
    
    dnp.plot.image(image)
    
    parameters = [centres, a, b, values]
    
    return image, parameters