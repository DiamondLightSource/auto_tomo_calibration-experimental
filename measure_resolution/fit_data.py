import numpy as np
import pylab as pl
from scipy.optimize import leastsq
from lmfit import minimize, Parameters, Parameter,\
                    report_fit, Model
from lmfit.models import PolynomialModel


def polynomial(y, p):
    c0, c1, c2, c3, c4, c5 = p
    
    return c0 + c1*y + c2*(y**2) + c3*(y**3) + c4*(y**4) + c5*(y**5)


def MTF(Y, X):
    """
    Fit a polynomial to the MTF curve
    """
    poly_mod = PolynomialModel(6)
     
    pars = poly_mod.guess(Y, x=X)
    model = poly_mod
     
    result = model.fit(Y, pars, x=X)
    # write error report
    print result.fit_report()
     
    c0 = result.best_values['c0']
    c1 = result.best_values['c1']
    c2 = result.best_values['c2']
    c3 = result.best_values['c3']
    c4 = result.best_values['c4']
    c5 = result.best_values['c5']
    
    params = [c0, c1, c2, c3, c4, c5]
    
    resolution = polynomial(10., params)
    
    
    return result.best_fit, resolution 