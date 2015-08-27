import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors


def calc_R(x, y, z, xc, yc, zc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2 + (z-zc)**2)


def f(c, x, y, z):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, z, *c)
    return Ri - Ri.mean()


def fitfunc(p, x, y, z):
    x0, y0, z0, R = p
    return ((x-x0)**2 + (y-y0)**2 + (z-z0)**2)


def leastsq_sphere(x, y, z, rad, centr):
    # coordinates of the barycenter
    x_m, y_m, = centr
    z_m = np.mean(z)

    p0 = [x_m, y_m, z_m, rad]
    errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[3]**2
    
    p1, cov, infodict, mesg, ier = optimize.leastsq(errfunc, p0, args=(x, y, z) ,full_output=True)
    
    ss_err=(infodict['fvec']**2).sum()
    ss_tot=((y-np.mean(y))**2).sum()
    rsquared=1-(ss_err/ss_tot)
    
    return p1, rsquared
