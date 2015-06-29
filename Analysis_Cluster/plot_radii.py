import numpy as np
import pylab as pl


radii = []
for i in range(0,360,10):
    radii.append(np.load('/dls/tmp/jjl36382/radii/radii%03i.npy' % i))
    
        
radii_np = np.zeros((360,181))
for i in range(36):
    radii_np[i*10:i*10+10,:] = radii[i]

# Remove the anomalous radii
radii_med = np.median(radii_np)
one_std_dev = np.std(radii_np)
for i in xrange(0,360):
    for j in xrange(0,181):
        if abs(radii_np[i, j] - radii_med) > one_std_dev:
            print abs(radii_np[i, j] - radii_med)
            radii_np[i, j] = radii_med
            

# Plot

pl.imshow(radii_np.T)
pl.title(r'Radii of real sphere as a function of 2 spherical angles $\theta$ and $\phi$',\
         fontdict={'fontsize': 16,'verticalalignment': 'bottom','horizontalalignment': 'center'})
pl.xlabel(r'$\theta$', fontdict={'fontsize': 14,'verticalalignment': 'top','horizontalalignment': 'center'})
pl.ylabel(r'$\phi$', fontdict={'fontsize': 14,'verticalalignment': 'bottom','horizontalalignment': 'right'}, rotation=0)
#pl.xticks(np.arange(0, 360, 10), theta_bord)
#pl.yticks(np.arange(0, len(phi_bord)+1, 10), phi_bord)
pl.colorbar(shrink=0.8)
pl.savefig("/dls/tmp/jjl36382/median_repair.png")

pl.show()
