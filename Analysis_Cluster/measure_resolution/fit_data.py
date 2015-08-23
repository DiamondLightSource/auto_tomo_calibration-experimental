import numpy as np
import pylab as pl
from lmfit import minimize, Parameters, Parameter,\
                    report_fit, Model
from lmfit.models import StepModel,PolynomialModel, ExponentialModel, DonaichModel, GaussianModel, LorentzianModel, ConstantModel, RectangleModel, LinearModel


def gaussian(x, height, center, width, offset):
    return (height/np.sqrt(2*np.pi)*width) * np.exp(-(x - center)**2/(2*width**2)) + offset

def polynomial(c0,c1,c2,c3,c4,c5,c6,y):
    return c0 + c1*y + c2*(y**2) + c3*(y**3) + c4*(y**4) + c5*(y**5)+ c6*(y**6)
    

def MTF(Y, X):
    """
    Fit a polynomial to the MTF curve
    """
    lin_mod = LinearModel(prefix="line_")
    exp_mod = ExponentialModel()
    const_mod = ConstantModel()
    poly_mod = PolynomialModel(7)
    
    #X = list(reversed(X))
    
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
    c6 = result.best_values['c6']
    
    limit = polynomial(c0,c1,c2,c3,c4,c5,c6,9)
    return result.best_fit, limit




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
        
        gauss_mod.set_param_hint('width', value = stdev / 2., min=stdev / 3., max=stdev)
        gauss_mod.set_param_hint('fwhm', expr='2.3548*width')
        pars = gauss_mod.make_params(height=amp, center=centre, width=stdev / 2., offset=offset)
        
        pars += step_mod.guess(Y, x=X, center=centre)

        pars += const_mod.guess(Y, x=X)
        
        pars['width'].vary = False
        
        mod = const_mod + gauss_mod + step_mod
        result = mod.fit(Y, pars, x=X)
        # write error report
        #print result.fit_report()
        
        fwhm = result.best_values['width'] * 2.3548
        
    return X, result.best_fit, result.redchi, fwhm


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
        
    return X, result.best_fit, result.redchi, 0


def Rectangle(signal, guess):
    
    if guess == False:
        return [0, 0, 0]
    else:
        amp, centre, stdev, offset = guess
        
        data = np.array([range(len(signal)), signal]).T
        X = data[:,0]
        Y = data[:,1]

        step_mod = Rectangle()
        const_mod = ConstantModel()
        
        pars = step_mod.guess(Y, x=X, center=centre)
        pars += const_mod.guess(Y, x=X)

        mod = step_mod + const_mod
        result = mod.fit(Y, pars, x=X)
        # write error report
        #print result.fit_report()
        fwhm = result.best_values['sigma'] * 2.3548
        print fwhm
        
    return X, result.best_fit, result.redchi, 0


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
        
        pars['gauss_sigma'].vary = False
        mod = gauss_mod + const_mod
        result = mod.fit(Y, pars, x=X)
        # write error report
        #print result.fit_report()
        fwhm = result.best_values['gauss_sigma'] * 2.3548

        
    return X, result.best_fit, result.redchi, fwhm


def minimized_residuals(signal, guess):
    if guess == False:
        return [0,0]
    else:
        
        X1, result1, err1, fwhm1 = GaussConst(signal, guess)
        X2, result2, err2, fwhm2 = GaussStepConst(signal, guess)
        X3, result3, err3, fwhm3 = Step(signal, guess)
        
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
            print "GaussConst"
            return X1, result1, fwhm1
        elif err2 == np.min(errors):
            print "GaussConstStep!"
            return X2, result2, fwhm2
        elif err3 == np.min(errors):
            return X3, result3, fwhm3
    
    