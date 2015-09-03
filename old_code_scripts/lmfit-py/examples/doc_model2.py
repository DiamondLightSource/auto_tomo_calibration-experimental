#!/usr/bin/env python
#<examples/model_doc2.py>
from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import Model

import matplotlib.pyplot as plt

data = loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1] + 0.25*x - 1.0

def gaussian(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2))

def line(x, slope, intercept):
    "line"
    return slope * x + intercept

mod = Model(gaussian) + Model(line)
pars  = mod.make_params( amp=5, cen=5, wid=1, slope=0, intercept=1)

result = mod.fit(y, pars, x=x)

print(result.fit_report())

plt.plot(x, y,         'bo')
plt.plot(x, result.init_fit, 'k--')
plt.plot(x, result.best_fit, 'r-')
plt.show()
#<end examples/model_doc2.py>
