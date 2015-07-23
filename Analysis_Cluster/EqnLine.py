import numpy as np
import pylab as pl


def slope(pt1, pt2):
    """
    Calculate the slope
    """
    try:
        return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    except:
        # division by zero indicates no slope
        return 0


def interceipt(pt1, pt2):
    """
    Get interceipt
    y = mx + b
    """
    m = slope(pt1, pt2)
    b = pt1[1] - m * pt1[0]
    return b


def get_y(m, b, x):
    y = m * x + b
    return y


def eqn_line(image, m):
    """
    goes along a line given a certain interceipt
    takes in another interceipt and goes along it again 
    registering all the pixel values
    """
    
#     pl.imshow(image)
#     pl.show()

    x = range(-image.shape[0], image.shape[0])
        
    pixels = []
    # for each interceipt
    for b in x:
        # draw lines parallel to
        # the line connecting centres
        cop = image.copy()
        line = []
        for i in range(image.shape[0]):
            y = get_y(m, b, i)
            try:
                line.append(cop[i, y])
#                 cop[i, y] = 0
#                 pl.imshow(cop, cmap = "Greys_r")
#                 pl.pause(0.001)except:
            except:
                continue
            
        if line:
            pixels.append(line)
        
    for i in range(len(pixels)):
        pl.close('all')
        pl.plot(pixels[i])
        pl.pause(0.5)
    
    return
