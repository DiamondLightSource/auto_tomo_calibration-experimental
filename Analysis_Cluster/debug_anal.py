import numpy as np

centres = np.load('/home/jjl36382/auto_tomo_calibration-experimental/Analysis_Cluster/modified_centroids.npy')
N = len(centres)

print "centres at the beginning"
for i in range(N):
    print "I: ", i, " ", centres[i]
    
# Find the (slice, index) of the elements which have no
# near neighbour, but are located further in the array
bad_pair = []
for slice in range(0, N - 4):
    for centre_index in range(len(centres[slice])):
        centre = centres[slice][centre_index]
        for centre_next in centres[slice + 1]:
            # if the centres are within the tolerance
            # and they are neighbours then stop
            if np.allclose(np.asarray(centre), np.asarray(centre_next), 0, 20):
                break
        else:
            for centre_next2 in centres[slice + 2]:
                if np.allclose(np.asarray(centre), np.asarray(centre_next2), 0, 20):
                    break
            else:
                for centre_next3 in centres[slice + 3]:
                    if np.allclose(np.asarray(centre), np.asarray(centre_next3), 0, 20):
                        break
                else:
                    for centre_next4 in centres[slice + 4]:
                        if np.allclose(np.asarray(centre), np.asarray(centre_next4), 0, 20):
                            break
                    # if they are similar ant not neighbours
                    # then remove it from the list
                    else:
                        for slice2 in range(slice + 5, N):
                            for index in range(len(centres[slice2])):
                                centre_further = centres[slice2][index]
                                #print "pair ", centre_further, centre
                                if np.allclose(np.asarray(centre), np.asarray(centre_further), 0, 20):
                                    bad_pair.append((slice, centre_index))
#-----------------------------------------------------------

# Get unique pairs of the slices and indices to be removed
# go from top to bottom to preserve indices
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
    #     radius[slice][index] = []
        
    # If a list was completely wiped out
    # it will be empty, hence delete empty lists
    bad_indices = []
    N = len(centres)
    for n in range(N):
        for element in centres[n]:
            if not element:
                bad_indices.append(n)
            
    centres = np.delete(centres, bad_indices)
    # radius = np.delete(radius, bad_indices)

#-------------------------------------------------------------

N = len(centres)
print "after removing distant centres"
for i in range(N):
    print "I: ", i, " ", centres[i]

index_top = [] # indexes of tops of spheres
centres_top = [] # centres of tops of spheres
index_bot = [] # indexes of bottoms of spheres
centres_bot = [] # centres of bottoms of spheres
centres_picked = [] # to store the (x,y) centres of spheres
radius_top = [] # Mimick the centre deletion process
radius_bot = []

# If a slice does not have neighbours within 4 slices
# the it is a bug
for slice in range(0, N-4):
    # Pick centres
    for centre in centres[slice]:
        # If the centre is not in the two following slices (to prevent from bugs): bottom of sphere
        for centre_1 in centres[slice+1]:
            # if centres equal to within about 10 pixels
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
                    # slices then add them to centre bot
                    else:
                        index_bot.append(slice)
                        centres_bot.append(centre)
#                 radius_bot.append(radius[slice])
        
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

# FIX IF THE CENTRES STARTED ACCUMULATING NEAR THE END
# AND THEN THE SLICES ENDED



"""for slice in range(N-4,N-1):
    # Pick centres
    for centre in centres[slice]:
        # If the centre is not in the two following slices (to prevent from bugs): bottom of sphere
        for centre_fol in centres[slice+1]:
            # if centres equal to within about 10 pixels
            if np.allclose(np.asarray(centre), np.asarray(centre_fol), 0, 20): 
                break
        else:
            index_bot.append(slice)
            centres_bot.append(centre)
#             radius_bot.append(radius[slice])
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
#             radius_top.append(radius[slice])"""

# for centre in centres[N-1]:
#     index_bot.append(N-1)
#     centres_bot.append(centre)
#     radius_bot.append(radius[N-1])
"""            
print len(index_bot)
print len(index_top)"""

# temp_cent = []
# for bot in centres_bot:
#     element_top = [top for top in centres_top if np.allclose(np.asarray(bot), np.asarray(top), 0, 10)]
#     if element_top:
#         print element_top
#         temp_cent.append(element_top[0])
# 
# centres_top = temp_cent

# Remove bugs = wrong centres that were detected only once
index_top_del = []
index_bot_del = []
# for i_top in range(len(index_top)):
#     if (index_top[i_top] in index_bot) and (centres_top[i_top] in centres_bot): # bugs are the only centres that appear once in both lists of edges
#         print index_top[i_top]
#         print centres_top[i_top]
#         i_bot = centres_bot.index((centres_top[i_top]))
#         index_top_del.append(i_top)
#         index_bot_del.append(i_bot)

        
# index_top[:] = [item for i,item in enumerate(index_top) if i not in index_top_del]
# centres_top[:] = [item for i,item in enumerate(centres_top) if i not in index_top_del]
# index_bot[:] = [item for i,item in enumerate(index_bot) if i not in index_bot_del]
# centres_bot[:] = [item for i,item in enumerate(centres_bot) if i not in index_bot_del]
# radius_top[:] = [item for i,item in enumerate(radius_top) if i not in index_top_del]
# radius_bot[:] = [item for i,item in enumerate(radius_bot) if i not in index_bot_del]


# remove the padded lists
del centres_bot[-1]
del centres_top[-1]
# del radius_bot[-1]
# del radius_top[-1]
del index_top[-1] 
del index_bot[-1]
centres = np.delete(centres, -1)
centres = np.delete(centres, -1)
centres = np.delete(centres, -1)
centres = np.delete(centres, -1)
centres = np.delete(centres, -1)
centres = np.delete(centres, -1)
# radius = np.delete(radius, -1)
# radius = np.delete(radius, -1)


print len(index_bot)
print len(index_top)
print len(centres_top)
print len(centres_bot)
print "after removing distant centres"

# N = len(centres)
# for i in range(N):
#     print "I: ", i, " ", centres[i]
                                 
print "remove centres which repeat only once"
1
1