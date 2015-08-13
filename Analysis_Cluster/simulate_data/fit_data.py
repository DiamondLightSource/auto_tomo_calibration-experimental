import numpy as np
import pylab as pl
from lmfit import minimize, Parameters, Parameter,\
                    report_fit, Model, CompositeModel
from lmfit.models import StepModel, GaussianModel, LorentzianModel, ConstantModel, RectangleModel


def gaussian(x, height, center, width, offset):
    return (height/np.sqrt(2*np.pi)*width) * np.exp(-(x - center)**2/(2*width**2)) + offset


def gauss_step_const(signal, guess):
    
    if guess == False:
        return [0, 0]
    else:
        amp, centre, stdev, offset = guess
        
        data = np.array([range(len(signal)), signal]).T
        X = data[:,0]
        Y = data[:,1]

#         gauss_mod = Model(gaussian)
        gauss_mod = GaussianModel(prefix='gauss')
        step_mod = StepModel(form='linear', prefix='step_')
        const_mod = ConstantModel(prefix="const_")
        
#         pars = gauss_mod.make_params(height=amp, center=centre, width=stdev / 3., offset=offset)
        pars = gauss_mod.make_params(amplitude=amp, center=centre, sigma=stdev / 3.)
        gauss_mod.set_param_hint('sigma', value = stdev / 3., min=stdev / 2., max=stdev)
        
        pars += step_mod.make_params(center=centre, vary=False)
        pars += const_mod.guess(Y, x=X)
        
        
        
        mod = step_mod + gauss_mod + const_mod
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


def gaussFN_const(signal, guess):
    
    if guess == False:
        return [0, 0]
    else:
        amp, centre, stdev, offset = guess
        
        data = np.array([range(len(signal)), signal]).T
        X = data[:,0]
        Y = data[:,1]

        gauss_mod = GaussianModel(prefix='gauss_')
        const_mod = ConstantModel(prefix='const_')
        
        #pars = lorentz_mod.make_params(amplitude=amp, center=centre, sigma=stdev / 3.)
        #lorentz_mod.set_param_hint('sigma', value = stdev / 3., min=0., max=stdev)
        
        pars = gauss_mod.guess(Y, x=X, center=centre, sigma=stdev / 3., amplitude=amp)
        #pars += step_mod.guess(Y, x=X, center=centre)
        pars += const_mod.guess(Y, x=X)

        mod = gauss_mod + const_mod
        result = mod.fit(Y, pars, x=X)
        # write error report
        print result.fit_report()
    
    return X, result.best_fit


def minimized_residuals(signal, guess):
    if guess == False:
        return [0, 0]
    else:
        
        X1, result1 = gaussFN_const(signal, guess)
        X2, result2 = gauss_step_const(signal, guess)
        
    return result1, result2

    