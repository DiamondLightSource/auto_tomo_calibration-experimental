import numpy as np
import pylab as pl
from lmfit import minimize, Parameters, Parameter,\
                    report_fit, Model
from lmfit.models import StepModel, GaussianModel, LorentzianModel, ConstantModel, RectangleModel


def gaussian(x, height, center, width, offset):
    return (height/np.sqrt(2*np.pi)*width) * np.exp(-(x - center)**2/(2*width**2)) + offset


def gauss_step(signal, guess):
    
    if guess == False:
        return [0, 0]
    else:
        amp, centre, stdev, offset = guess
        
        data = np.array([range(len(signal)), signal]).T
        X = data[:,0]
        Y = data[:,1]

        gauss_mod = Model(gaussian)
        const_mod = ConstantModel(prefix='const_')
        
        pars = gauss_mod.make_params(height=amp, center=centre, width=stdev / 3., offset=offset)
        #gauss_mod.set_param_hint('width', value = stdev / 3., min=0., max=stdev)
        
        pars += const_mod.guess(Y, x=X)
        
        mod = const_mod + gauss_mod
        result = mod.fit(Y, pars, x=X)
        # write error report
        #print result.fit_report()
    
    return X, result.best_fit


def box_step(signal, guess):
    
    if guess == False:
        return [0, 0]
    else:
        amp, centre, stdev, offset = guess
        
        data = np.array([range(len(signal)), signal]).T
        X = data[:,0]
        Y = data[:,1]

        box_mod = RectangleModel(prefix='rect_')
        const_mod = ConstantModel(prefix='const_')
        
        pars = box_mod.guess(Y, x=X, amplitude=amp)
        pars += const_mod.guess(Y, x=X)

        mod = box_mod + const_mod
        result = mod.fit(Y, pars, x=X)
        # write error report
        #print result.fit_report()
    
    return X, result.best_fit


def lorentz_step(signal, guess):
    
    if guess == False:
        return [0, 0]
    else:
        amp, centre, stdev, offset = guess
        
        data = np.array([range(len(signal)), signal]).T
        X = data[:,0]
        Y = data[:,1]

        lorentz_mod = LorentzianModel(prefix='lorentz_')
        step_mod = StepModel(form='linear', prefix='step_')
        const_mod = ConstantModel(prefix='const_')
        
        #pars = lorentz_mod.make_params(amplitude=amp, center=centre, sigma=stdev / 3.)
        #lorentz_mod.set_param_hint('sigma', value = stdev / 3., min=0., max=stdev)
        
        pars = lorentz_mod.guess(Y, x=X, center=centre, sigma=stdev / 3., amplitude=amp)
        #pars += step_mod.guess(Y, x=X, center=centre)
        pars += const_mod.guess(Y, x=X)

        mod = lorentz_mod + const_mod
        result = mod.fit(Y, pars, x=X)
        # write error report
        #print result.fit_report()
    
    return X, result.best_fit

