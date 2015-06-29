import numpy as np

radii = []
for i in range(0,360,10):
    radii.append(np.load('/dls/tmp/jjl36382/radii/radii%03i.npy' % i))
        
radii_np = np.zeros((360,181))
for i in range(36):
    radii_np[i*10:i*10+10,:] = radii[i]
    
print np.mean(radii_np)
# Remove the anomalous radii
radii_avg = np.mean(radii_np)
one_std_dev = np.std(radii_np)

print one_std_dev

for i in xrange(0,360,10):
    for j in xrange(0,181):
        if abs(radii_np[i, j] - radii_avg) >= one_std_dev:
            radii_np[i, j] = radii_avg
            
print np.mean(radii_np)
print np.std(radii_np)
