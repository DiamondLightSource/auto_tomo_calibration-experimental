import pickle
import pylab as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#cd /dls/tmp/jjl36382/results

pl.close('all')

if __name__ == '__main__' :
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("-a", "--start",
                         dest="a",
                         help="Starting position of a file",
                         default=500,
                         type='int')
    parser.add_option("-b", "--end",
                        dest="b",
                        help="End position of a file",
                        default=500,
                        type='int')
    parser.add_option("-c", "--step",
                        dest="c",
                        help="Step size",
                        default=500,
                        type='int')


    (options, args) = parser.parse_args()

    start = options.a
    stop = options.b 
    step = options.c
    input_filename = args[0]
    
    start = start + step
    stop = stop - step

fig = pl.figure()
ax = fig.gca(projection='3d')

# ---------------------------------------- Get data -----------------------------------------
"""
Loads all of the detected parameters of segmented circles
in a slice and appends them to a list.
"""
data = []
for i in range(start, stop, step):
    f = open(input_filename % i, 'r')
    data.append(pickle.load(f))
    f.close()

N = len(data)

bord_circles = []
centroids_sphere = []
radii_circles = []
perimeters = []

for i in range(N):
    bord_circles.append(data[i][0])
    centroids_sphere.append(data[i][1])
    radii_circles.append(data[i][2])
    perimeters.append(data[i][3])

print bord_circles
print centroids_sphere
print radii_circles

# Plot

#for slice in range(N):
#    for i in range(len(perimeters[slice])):
#        ax.plot(perimeters[slice][i][0] + bord_circles[slice][i][0], perimeters[slice][i][1] + bord_circles[slice][i][2], slice*10+10)

#ax.set_xlim(0,2560)
#ax.set_ylim(0,2560)
#ax.set_zlim(0,2560)
#pl.title('Sphere detection on real image')

#pl.show()

# ------------ Sort out spheres for radii_angles (i.e. sort out centres + radii) ------------

# Calculate centres according to the whole image

centres = []
for slice in range(N):
    cxcy = []
    for i in range(len(centroids_sphere[slice])):
        cx = centroids_sphere[slice][i][0] + bord_circles[slice][i][0]
        cy = centroids_sphere[slice][i][1] + bord_circles[slice][i][2]
        cxcy.append((cx,cy))
    #cxcy = np.asarray(cxcy)
    centres.append(cxcy)

for slice in range(N):
    print slice, centres[slice]

# Remove wrong areas

# Find the edges of spheres
# Initialisation
index_top = [] # indexes of tops of spheres
centres_top = [] # centres of tops of spheres
index_bot = [] # indexes of bottoms of spheres
centres_bot = [] # centres of bottoms of spheres
centres_picked = [] # to store the (x,y) centres of spheres
for centre in centres[0]:
    centres_picked.append(centre)
    index_top.append(0)
    centres_top.append(centre)
# Process
for slice in range(1,N-2):
    # Pick centres
    for centre in centres[slice]:
        # If the centre is not in the two following slices (to prevent from bugs): bottom of sphere
        for centre_fol in centres[slice+1]:
            if np.allclose(np.asarray(centre), np.asarray(centre_fol), 0, 10): # if centres equal to within about 10 pixels
                break
        else:
            for centre_foll in centres[slice+2]:
                if np.allclose(np.asarray(centre), np.asarray(centre_foll), 0, 10):
                    break
            else:
                index_bot.append(slice)
                centres_bot.append(centre)
        # If the centre has not been picked before: top of sphere
        for centre_p in centres_picked:
            if np.allclose(np.asarray(centre), np.asarray(centre_p), 0, 10):
                break
        else:
            centres_picked.append(centre)
            index_top.append(slice)
            centres_top.append(centre)

for centre in centres[N-1]:
    index_bot.append(N-1)
    centres_bot.append(centre)

# Remove bugs = wrong centres that were detected only once
index_top_del = []
index_bot_del = []
for i_top in range(len(index_top)):
    if (index_top[i_top] in index_bot) and (centres_top[i_top] in centres_bot): # bugs are the only centres that appear once in both lists of edges
        i_bot = centres_bot.index((centres_top[i_top]))
        index_top_del.append(i_top)
        index_bot_del.append(i_bot)
index_top[:] = [item for i,item in enumerate(index_top) if i not in index_top_del]
centres_top[:] = [item for i,item in enumerate(centres_top) if i not in index_top_del]
index_bot[:] = [item for i,item in enumerate(index_bot) if i not in index_bot_del]
centres_bot[:] = [item for i,item in enumerate(centres_bot) if i not in index_bot_del]

# Get (x,y) coordinates of centres of spheres

if np.allclose(np.asarray(centres_bot), np.asarray(centres_top), 0, 10):
    centres_zipped = np.asarray(zip(centres_top,centres_bot))
    centres_xy = np.array(np.median(centres_zipped, axis=1), dtype='int64')

# Calculate z coordinate of centres of spheres

if np.allclose(np.asarray(centres_bot), np.asarray(centres_top), 0, 10):
    edges = np.asarray(zip(index_top, index_bot))
    centres_z = np.array(np.median(edges, axis=1), dtype='int64')*10 + 10

centroids = zip(centres_xy[:,0], centres_xy[:,1], centres_z)

print 'Nb of spheres:', len(centroids)
print 'Centres of spheres:', centroids

# Sort out the 2D areas among spheres

nb_spheres = len(centroids)

slices_spheres = []
bords_spheres = []
radii_slices = []

for n in range(nb_spheres):
    
    slices = []
    bords = []
    radii = []
    
    for slice in range(edges[n][0], edges[n][1]+1):
        
        for i in range(len(centres[slice])):
            #TO BE CHANGED TO FIT DATA
            if (abs(centres[slice][i][0] - centroids[n][0]) < 10)\
            and (abs(centres[slice][i][1] - centroids[n][1]) < 10):
                slices.append(slice)
                bords.append(bord_circles[slice][i])
                radii.append(radii_circles[slice][i])
    
    if len(slices) > 10:
        slices_spheres.append(slices)
        bords_spheres.append(bords)
        radii_slices.append(radii)

for n in range(nb_spheres):
    print 'Len(areas[', n, ']):', len(areas_spheres[n])

for n in range(nb_spheres):
    print len(slices_spheres[n])
    print slices_spheres[n]
    print radii_slices[n]

# ---------------------------------- Save centres + radii -----------------------------------
# --------------------------------- (write data in a file) ----------------------------------

print("Saving data")

f = open('centres.txt', 'w')
f.write('Centres of the spheres')
for i in range(nb_spheres):
    f.write(repr(centroids[i]) + '\n')
f.close()

f = open('radii.txt', 'w')
f.write('Radii of the spheres: [horizontal, vertical]')
for i in range(nb_spheres):
    f.write(repr(radii_slices[i]) + '\n')
f.close()