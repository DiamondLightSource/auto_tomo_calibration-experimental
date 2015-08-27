import lmfit
import numpy as np
import matplotlib
# matplotlib.use('WXAgg')

import matplotlib.pyplot as plt

x = np.linspace(1, 10, 250)
np.random.seed(0)
y = 3.0*np.exp(-x/2) -5.0*np.exp(-(x-0.1)/10.) + 0.1*np.random.randn(len(x))

p = lmfit.Parameters()
p.add_many(('a1', 4.), ('a2', 4.), ('t1', 3.), ('t2', 3.))

def residual(p):
   v = p.valuesdict()
   return v['a1']*np.exp(-x/v['t1']) + v['a2']*np.exp(-(x-0.1)/v['t2'])-y

# create Minimizer
mini = lmfit.Minimizer(residual, p)

# first solve with Nelder-Mead
out1 = mini.minimize(method='Nelder')

# then solve with Levenberg-Marquardt using the
# Nelder-Mead solution as a starting point
out2 = mini.minimize(method='leastsq', params=out1.params)

lmfit.report_fit(out2.params, min_correl=0.5)

ci, trace = lmfit.conf_interval(mini, out2, sigmas=[0.68,0.95],
                                trace=True, verbose=False)
lmfit.printfuncs.report_ci(ci)

plot_type = 2
if plot_type == 0:
    plt.plot(x, y)
    plt.plot(x, residual(out2.params)+y )

elif plot_type == 1:
    cx, cy, grid = lmfit.conf_interval2d(mini, out2, 'a2','t2',30,30)
    plt.contourf(cx, cy, grid, np.linspace(0,1,11))
    plt.xlabel('a2')
    plt.colorbar()
    plt.ylabel('t2')

elif plot_type == 2:
    cx, cy, grid = lmfit.conf_interval2d(mini, out2, 'a1','t2',30,30)
    plt.contourf(cx, cy, grid, np.linspace(0,1,11))
    plt.xlabel('a1')
    plt.colorbar()
    plt.ylabel('t2')


elif plot_type == 3:
    cx1, cy1, prob = trace['a1']['a1'], trace['a1']['t2'],trace['a1']['prob']
    cx2, cy2, prob2 = trace['t2']['t2'], trace['t2']['a1'],trace['t2']['prob']
    plt.scatter(cx1, cy1, c=prob, s=30)
    plt.scatter(cx2, cy2, c=prob2, s=30)
    plt.gca().set_xlim((2.5, 3.5))
    plt.gca().set_ylim((11, 13))
    plt.xlabel('a1')
    plt.ylabel('t2')

if plot_type > 0:
    plt.show()
