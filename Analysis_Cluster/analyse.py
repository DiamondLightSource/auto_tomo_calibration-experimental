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

"""
Store the borders, centroids, radii and perimeters
"""
for i in range(N):
    bord_circles.append(data[i][0])
    centroids_sphere.append(data[i][1])
    radii_circles.append(data[i][2])
    perimeters.append(data[i][3])


f = open('/home/jjl36382/auto_tomo_calibration-experimental/Analysis_Cluster/original_centroids.txt', 'w')
f.write(repr(centroids_sphere))
f.close()

# Remove empty lists---------------------------------------------------------
bad_indices = []
for n in range(N):
    empt = centroids_sphere[n]
    if not empt:
        bad_indices.append(n)
        
centroids_sphere = np.delete(centroids_sphere, bad_indices)
bord_circles = np.delete(bord_circles, bad_indices)
radii_circles = np.delete(radii_circles, bad_indices)
perimeters = np.delete(perimeters, bad_indices)
#--------------------------------------------------------

N = len(perimeters)


# fig = pl.figure()
# ax = fig.gca(projection='3d')
# for slice in range(N):
#     for i in range(len(perimeters[slice])):
#         ax.plot(perimeters[slice][i][0] + bord_circles[slice][i][0], perimeters[slice][i][1] + bord_circles[slice][i][2], slice*step+step)
# 
# 
# ax.set_xlim(0, stop)
# ax.set_ylim(0, stop)
# ax.set_zlim(0, stop)
# pl.title('Sphere detection on real image')
# pl.savefig("/dls/tmp/jjl36382/analysis/reconstruction.png")
# pl.show()

# ------------ Sort out spheres for radii_angles (i.e. sort out centres + radii) ------------

# Calculate centres according to the whole image
"""
An element of centres array is an array of tuples
where each array stores slice information
and tuples store the centres of the segmented circles
of the slice
"""

# Remove repeating centres--------------------------------------------------------
centres = []
radius = []
for slice in range(N):
    cxcy = []
    pair = []
    r = []
    for i in range(len(centroids_sphere[slice])):
        cx = centroids_sphere[slice][i][0] + bord_circles[slice][i][0]
        cy = centroids_sphere[slice][i][1] + bord_circles[slice][i][2]
        rad = radii_circles[slice][i]
        # IF THERE ARE SAME CENTRES IN THE SAME SLICE THEN
        # WE WILL GET ERRORS - REMOVE THEM
        cxcy[:] = [item for item in cxcy if not np.allclose(np.asarray((cx,cy)), np.asarray(item), 0, 20)]
        r[:] = [rad for item in cxcy if not np.allclose(np.asarray((cx,cy)), np.asarray(item), 0, 20)]
        # MODIFY THE RADIUS ACCORDINGLY
        r.append((rad))
        cxcy.append((cx,cy))
    #cxcy = np.asarray(cxcy)
    centres.append(cxcy)
    radius.append(r)
#--------------------------------------------------------


# PAD THE END OF THE LIST IN ORDER TO CHECK THROUGH
# EVERY ELEMENT IN THE NEXT STEP EASILY
#--------------------------------------------------------
centres.append([(1,1)])
centres.append([(1,1)])
centres.append([(1,1)])
centres.append([(1,1)])
centres.append([(1,1)])
centres.append([(1,1)])
radius.append([(666)])
radius.append([(666)])
radius.append([(666)])
radius.append([(666)])
radius.append([(666)])
radius.append([(666)])
N = len(centres)
#--------------------------------------------------------

# Load data --------------------------------------------------------
np.save('/home/jjl36382/auto_tomo_calibration-experimental/Analysis_Cluster/modified_centroids.npy', centres)
np.save('/home/jjl36382/auto_tomo_calibration-experimental/Analysis_Cluster/radius.npy', radius)


# Find the (slice, index) of the elements which have no
# near neighbour, but are located further in the array
#--------------------------------------------------------------------
bad_pair = []
for slice in range(0, N - 2):
    for centre_index in range(len(centres[slice])):
        centre = centres[slice][centre_index]
        for centre_next in centres[slice + 1]:
            # if the centres are within the tolerance
            # and they are neighbours then stop
            if np.allclose(np.asarray(centre), np.asarray(centre_next), 0, 15):
                break
        else:
            for centre_next2 in centres[slice + 2]:
                if np.allclose(np.asarray(centre), np.asarray(centre_next2), 0, 15):
                    break
            # if they are similar ant not neighbours
            # then remove it from the list
            else:
                bad_pair.append((slice, centre_index))
#                 for slice2 in range(slice + 3, N):
#                     for index in range(len(centres[slice2])):
#                         centre_further = centres[slice2][index]
#                         #print "pair ", centre_further, centre
#                         if np.allclose(np.asarray(centre), np.asarray(centre_further), 0, 15):
#                             bad_pair.append((slice, centre_index))

# Get unique pairs of the slices and indices to be
# removed
if bad_pair:
    unique_pairs = []
    unique_pairs.append(bad_pair[0])
    for index in range(1, len(bad_pair)):
        if bad_pair[index] in unique_pairs:
            continue
        else:
            unique_pairs.append(bad_pair[index])

    # Remove the indices slices and elements in them
    # which are anomalous
    centres = np.asarray(centres)
    sorted_by_slice = sorted(unique_pairs, key=lambda tup: tup[0])
    rev_unique = list(reversed(sorted_by_slice))
    
    # print "unique slices for deletion and reversed"
    # print rev_unique
    for i in rev_unique:
        slice = i[0]
        index = i[1]
        centres[slice][index] = []
        radius[slice][index] = []
        
    # If a list was completely wiped out
    # it will be empty, hence delete empty lists
    bad_indices = []
    N = len(centres)
    for n in range(N):
        for element in centres[n]:
            if not element:
                bad_indices.append(n)
            
    centres = np.delete(centres, bad_indices)
    radius = np.delete(radius, bad_indices)
#--------------------------------------------------------------------



#--------------------------------------------------------------------
# Find the centres and how much do they span (their size)
N = len(centres)
index_top = [] # indexes of tops of spheres
centres_top = [] # centres of tops of spheres
index_bot = [] # indexes of bottoms of spheres
centres_bot = [] # centres of bottoms of spheres
centres_picked = [] # to store the (x,y) centres of spheres
radius_top = [] # Mimick the centre deletion process
radius_bot = []

for slice in range(0, N-4):
    for centre in centres[slice]:
        
        # See if the centres appear within 4 slices
        for centre_1 in centres[slice+1]:
            # if centres equal to within about 20 pixels
            if np.allclose(np.asarray(centre), np.asarray(centre_1), 0, 20): 
                break
        else:
            for centre_2 in centres[slice+2]:
                if np.allclose(np.asarray(centre), np.asarray(centre_2), 0, 20):
                    break
            else:
                for centre_3 in centres[slice+3]:
                    if np.allclose(np.asarray(centre), np.asarray(centre_3), 0, 20):
                        break
                else:
                    for centre_4 in centres[slice+4]:
                        if np.allclose(np.asarray(centre), np.asarray(centre_4), 0, 20):
                            break
                    # if centres are not the same within four neighbouring
                    # slices then add them to centre_bot
                    else:
                        index_bot.append(slice)
                        centres_bot.append(centre)
#                         radius_bot.append(radius[slice])
                        
        # If the centre has not been picked before: top of sphere
        # If the neighbouring slices have centres close together
        # then add them to centres picked and top arrays
        for centre_p in centres_picked:
            if np.allclose(np.asarray(centre), np.asarray(centre_p), 0, 20):
                break
        else:
            centres_picked.append(centre)
            index_top.append(slice)
            centres_top.append(centre)
#             radius_top.append(radius[slice])

# This fixes the end of the list
for centre in centres[N-1]:
    index_bot.append(N-1)
    centres_bot.append(centre)
#     radius_bot.append(radius[N-1])

print centres_top
print centres_bot
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
# radius_top[:] = [item for i,item in enumerate(radius_top) if i not in index_top_del]
# radius_bot[:] = [item for i,item in enumerate(radius_bot) if i not in index_bot_del]


# remove the padded lists
#--------------------------------------------------------------------
del centres_bot[-1]
del centres_top[-1]
# del radius_bot[-1]
# del radius_top[-1]
del index_top[-1] 
del index_bot[-1]
centres = np.delete(centres, -1)
centres = np.delete(centres, -1)
radius = np.delete(radius, -1)
radius = np.delete(radius, -1)
centres = np.delete(centres, -1)
centres = np.delete(centres, -1)
radius = np.delete(radius, -1)
radius = np.delete(radius, -1)
centres = np.delete(centres, -1)
centres = np.delete(centres, -1)
radius = np.delete(radius, -1)
radius = np.delete(radius, -1)
#--------------------------------------------------------------------

print "indices and centres bot/top"                   
print len(index_bot)
print len(index_top)
print len(centres_top)
print len(centres_bot)

# SORT CENTRES_BOT AND CENTRES_TOP TO MATCH SIMILAR CENTRES
# THEN TAKE THE MEDIAN. IF THEY ARE NOT SORTER ALL_CLOSE DOES
# NOT WORK
# SORT ACCORDING TO THE CENTRES_TOP LIST
temp_cent = []
temp_rad = []
# for top in centres_top:
#     element_bot = [bot for bot in centres_bot if np.allclose(np.asarray(bot), np.asarray(top), 0, 10)]
#     element_rad = [radius_bot[index] for index, bot in enumerate(centres_bot) if np.allclose(np.asarray(bot), np.asarray(top), 0, 10)]
#     print element_bot
#     temp_cent.append(element_bot[0])
#     temp_rad.append(element_rad[0])
    
# Try sorting according to bot since top fails
# print centres_top
# for bot in centres_bot:
#     element_top = [top for top in centres_top if np.allclose(np.asarray(bot), np.asarray(top), 0, 10)]
#     element_rad = [radius_top[index] for index, top in enumerate(centres_top) if np.allclose(np.asarray(bot), np.asarray(top), 0, 10)]
#     if element_top:
#         print element_top
#         temp_cent.append(element_top[0])
#         temp_rad.append(element_rad[0])
# radius_bot = temp_rad
# centres_bot = temp_cent

# Get (x,y) coordinates of centres of spheres
print centres_top
print centres_bot
if np.allclose(np.asarray(centres_bot), np.asarray(centres_top), 0, 20):
    centres_zipped = np.asarray(zip(centres_top, centres_bot))
#     radius_zipped = np.asarray(zip(radius_top, radius_bot))
    centres_xy = np.array(np.median(centres_zipped, axis=1), dtype='int64')

# Get the mean of the radius double values
# radius_mean = []
# for slice in radius_zipped:
#     print slice
#     radius_mean.append(np.mean(np.max(slice)))

# Calculate z coordinate of centres of spheres
if np.allclose(np.asarray(centres_bot), np.asarray(centres_top), 0, 20):
    edges = np.asarray(zip(index_top, index_bot))
    centres_z = np.array(np.median(edges, axis=1), dtype='int64')*step + step

centroids = zip(centres_xy[:,0], centres_xy[:,1], centres_z)

# Sort out the 2D areas among spheres

nb_spheres = len(centroids)

slices_spheres = []
bords_spheres = []
radii_slices = []
bad_indices = []
radii_spheres = []
old_nb = nb_spheres

print "end of processing"
print len(centroids)
# print len(radius_zipped)
# print radius_zipped
# print radius_mean

for n in range(nb_spheres):
     
    slices = []
    bords = []
    radii = []
     
    for slice in range(edges[n][0], edges[n][1]+1):
         
        for i in range(len(centres[slice])):
            slices.append(slice)
            radii.append(radius[slice][i])
        
    if len(slices) > 5:
        slices_spheres.append(slices)
        radii_spheres.append(radii)
    else:
        nb_spheres -= 1
        bad_indices.append(n)

new_centroids = []
new_radius = []
for i in range(old_nb):
    if i not in bad_indices:
        new_centroids.append(centroids[i])
#         new_radius.append(ra)

# coordinates are shifted due to filenames
# not starting form 0
new_centroids = centroids
z_coordinate = []
for i in range(len(new_centroids)):
    z_coordinate.append(new_centroids[i][2] + start)

# ---------------------------------- Save centres + radii -----------------------------------
# --------------------------------- (write data in a file) ----------------------------------

print("Saving data")

f = open('/dls/tmp/jjl36382/results/centres.txt', 'w')
for i in range(nb_spheres):
    f.write(repr(new_centroids[i]) + '\n')
f.close()

f = open('/dls/tmp/jjl36382/results/centresX.txt', 'w')
for i in range(nb_spheres):
    f.write(repr(new_centroids[i][0]) + '\n')
f.close()

f = open('/dls/tmp/jjl36382/results/centresY.txt', 'w')
for i in range(nb_spheres):
    f.write(repr(new_centroids[i][1]) + '\n')
f.close()

f = open('/dls/tmp/jjl36382/results/centresZ.txt', 'w')
for i in range(nb_spheres):
    f.write(repr(z_coordinate[i]) + '\n')
f.close()

# f = open('/dls/tmp/jjl36382/results/radii.txt', 'w')
# for i in range(nb_spheres):
#     f.write(repr(radii_slices[i]) + '\n')
# f.close()

# f = open('/dls/tmp/jjl36382/results/radii_max.txt', 'w')
# for i in range(nb_spheres):
#     f.write(repr(max(radii_slices[i])) + '\n')
# f.close()

f = open('/dls/tmp/jjl36382/results/radii_max.txt', 'w')
for i in range(nb_spheres):
    f.write(repr(int(max(radii_spheres[i]))) + '\n')
f.close()

f = open('/dls/tmp/jjl36382/results/nb_spheres.txt', 'w')
f.write(repr(nb_spheres))
f.close()
