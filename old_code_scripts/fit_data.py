import numpy as np
import pylab as pl
from scipy.optimize import leastsq
from lmfit import minimize, Parameters, Parameter,\
                    report_fit, Model
from lmfit.models import PolynomialModel, GaussianModel, ConstantModel, RectangleModel


def polynomial(y, p):
    c0, c1, c2, c3, c4, c5 = p    
    return c0 + c1*y + c2*(y**2) + c3*(y**3) + c4*(y**4) + c5*(y**5)


def gaussian(x, amplitude, center, sigma, offset):
    return (amplitude/np.sqrt(2*np.pi)*sigma) * np.exp(-(x - center)**2/(2*sigma**2)) + offset


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


def GaussConst(signal, guess):
    
    amp, centre, stdev, offset = guess
    
    data = np.array([range(len(signal)), signal]).T
    X = data[:,0]
    Y = data[:,1]

    gauss_mod = GaussianModel(prefix='gauss_')
    const_mod = ConstantModel(prefix='const_')
    
    pars = gauss_mod.make_params(center=centre, sigma=stdev, amplitude=amp)
    pars += const_mod.guess(Y, x=X)
    pars['gauss_center'].min = centre - 5.
    pars['gauss_center'].max = centre + 5.
    pars['gauss_sigma'].max = stdev + 5.
    
    mod = gauss_mod + const_mod
    result = mod.fit(Y, pars, x=X)
    
    fwhm = result.best_values['gauss_sigma'] #* 2.3548
    centr = result.best_values['gauss_center']
    
    # Values within two stdevs i.e. 95%
    pl.plot(np.repeat(centr - fwhm * 2, len(Y)),
            np.arange(len(Y)), 'b-')
    pl.plot(np.repeat(centr + fwhm * 2, len(Y)),
            np.arange(len(Y)), 'b-')
    
    return X, result.best_fit, result.best_values['gauss_sigma'] * 4


def Box(signal, guess):
    
    amp, centre, stdev, offset = guess
    
    data = np.array([range(len(signal)), signal]).T
    X = data[:,0]
    Y = data[:,1]

    gauss_mod = RectangleModel(prefix='gauss_', mode='logistic')
    const_mod = ConstantModel(prefix='const_')
    
    pars = gauss_mod.make_params( center1=centre-stdev*3, center2=centre+stdev*3, sigma1=0, sigma2=0, amplitude=amp)
    pars += const_mod.guess(Y, x=X)
    pars['gauss_center1'].min = centre-stdev*3 - 3
    pars['gauss_center2'].max = centre-stdev*3 + 3
    pars['gauss_center2'].min = centre+stdev*3 - 3
    pars['gauss_center2'].max = centre+stdev*3 + 3
    
    mod = gauss_mod + const_mod
    result = mod.fit(Y, pars, x=X)
    
    c1 = result.best_values['gauss_center1']
    c2 = result.best_values['gauss_center2']
    
    pl.legend()
    
    return X, result.best_fit, c2-c1


def GaussManual(signal, guess):
    """
    Fits high contrast data very well
    """
    amp, centre, stdev, offset = guess
    
    data = np.array([range(len(signal)), signal]).T
    X = data[:,0]
    Y = data[:,1]

    gauss_mod = Model(gaussian)
    
    pars = gauss_mod.make_params(center=centre, sigma=stdev, amplitude=amp, offset=offset)
    pars['center'].vary = False
    pars['sigma'].min = stdev
    pars['sigma'].max = stdev + 5.
    
    mod = gauss_mod
    result = mod.fit(Y, pars, x=X)
    fwhm = result.best_values['sigma'] * 2.3548

    return X, result.best_fit, result.best_values['sigma'] * 4