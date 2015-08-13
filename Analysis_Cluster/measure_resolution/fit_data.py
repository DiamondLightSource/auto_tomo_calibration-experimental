import numpy as np
import pylab as pl
from lmfit import minimize, Parameters, Parameter,\
                    report_fit, Model
from lmfit.models import StepModel,BreitWignerModel, DonaichModel, GaussianModel, LorentzianModel, ConstantModel, RectangleModel, LinearModel


def gaussian(x, height, center, width, offset):
    return (height/np.sqrt(2*np.pi)*width) * np.exp(-(x - center)**2/(2*width**2)) + offset


def Donaich(signal, guess):
    """
    Model used in photo emission
    """
    if guess == False:
        return [0, 0, 0]
    else:
        amp, centre, stdev, offset = guess
        
        data = np.array([range(len(signal)), signal]).T
        X = data[:,0]
        Y = data[:,1]

#         gauss_mod = Model(gaussian)
        don_mod = DonaichModel()
        
        pars = don_mod.make_params(amplitude=amp, center=centre, sigma=stdev / 3., gamma = 1)
        don_mod.set_param_hint('sigma', value = stdev / 2., min=0., max=stdev*2)
        
        result = don_mod.fit(Y, pars, x=X)
        # write error report
        #print result.fit_report()
    
    return X, result.best_fit, result.redchi


def Breit(signal, guess):
    """
    Model used in physics for assymetric curves
    """
    if guess == False:
        return [0, 0, 0]
    else:
        amp, centre, stdev, offset = guess
        
        data = np.array([range(len(signal)), signal]).T
        X = data[:,0]
        Y = data[:,1]

#         gauss_mod = Model(gaussian)
        breit_mod = BreitWignerModel()
        
        
        pars = breit_mod.make_params(amplitude=amp, center=centre, sigma=stdev / 3.)
        breit_mod.set_param_hint('sigma', value = stdev / 2., min=0., max=stdev*2)
        
        result = breit_mod.fit(Y, pars, x=X)
        # write error report
        #print result.fit_report()
    
    return X, result.best_fit, result.redchi



def GaussStepConst(signal, guess):
    """
    Fits high contrast data very well
    """
    if guess == False:
        return [0, 0, 0]
    else:
        amp, centre, stdev, offset = guess
        
        data = np.array([range(len(signal)), signal]).T
        X = data[:,0]
        Y = data[:,1]

#         gauss_mod = Model(gaussian)
        gauss_mod = Model(gaussian)
        const_mod = ConstantModel()
        step_mod = StepModel(prefix='step')
        
        pars = gauss_mod.make_params(height=amp, center=centre, width=stdev / 3., offset=offset)
#         pars = gauss_mod.make_params(amplitude=amp, center=centre, sigma=stdev / 3.)
        gauss_mod.set_param_hint('width', value = stdev / 3., min=stdev / 2., max=stdev)
        pars += step_mod.guess(Y, x=X, center=centre)

        pars += const_mod.guess(Y, x=X)
    
        
        mod = const_mod + gauss_mod + step_mod
        result = mod.fit(Y, pars, x=X)
        # write error report
        #print result.fit_report()
    return X, result.best_fit, result.redchi


def Step(signal, guess):
    
    if guess == False:
        return [0, 0, 0]
    else:
        amp, centre, stdev, offset = guess
        
        data = np.array([range(len(signal)), signal]).T
        X = data[:,0]
        Y = data[:,1]

        step_mod = StepModel(prefix='step')
        const_mod = ConstantModel(prefix='const_')
        
        pars = step_mod.guess(Y, x=X, center=centre)
        pars += const_mod.guess(Y, x=X)

        mod = step_mod + const_mod
        result = mod.fit(Y, pars, x=X)
        # write error report
        #print result.fit_report()
        
    return X, result.best_fit, result.redchi


def GaussConst(signal, guess):
    
    if guess == False:
        return [0, 0, 0]
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
        #print result.fit_report()
    return X, result.best_fit, result.redchi


def minimized_residuals(signal, guess):
    if guess == False:
        return [0,0]
    else:
        
        X1, result1, err1 = GaussConst(signal, guess)
        X2, result2, err2 = GaussStepConst(signal, guess)
        X3, result3, err3 = Step(signal, guess)
        
        errors = []
        if np.isnan(err1):
            pass
        else:
            errors.append(err1)
        if np.isnan(err2):
            pass
        else:
            errors.append(err2)
            
        if np.isnan(err3):
            pass
        else:
            errors.append(err3)
            
            
        if err1 == np.min(errors):
            return X1, result1
        elif err2 == np.min(errors):
            return X2, result2
        elif err3 == np.min(errors):
            return X3, result3
    
    