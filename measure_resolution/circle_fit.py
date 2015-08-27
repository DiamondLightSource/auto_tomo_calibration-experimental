import numpy as np
from scipy import optimize


def calc_R(x, y, xc, yc):
    """
    Calculate the distance of each 2D points
    from the center (xc, yc)
    """
    return np.sqrt((x - xc)**2 + (y - yc)**2)


def f(c, x, y):
    """
    Calculate the algebraic distance between the data
    points and the mean circle centered at c=(xc, yc)
    """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()


def leastsq_circle(x, y):
    """
    Uses least squares to fit a circle by using
    the co-ordinates of the edges.
    """
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x, y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    
    return xc, yc, R, residu
