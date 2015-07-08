centres = [[(568, 582), (592, 492), (686, 802)],
           [(780, 1200), (1049, 941)], # this element is found ages ago and hence influence results
           [(751, 421), (1007, 881)],
           [(967, 886)],
           [(967, 885)],
           [(967, 886)],
           [(967, 886)],
           [(838, 668), (967, 886)],
           [(838, 668), (967, 886)],
           [(838, 668), (967, 886)],
           [(838, 668), (967, 886)],
           [(838, 668), (967, 886)],
           [(838, 668), (967, 886)],
           [(838, 668), (967, 886)],
           [(838, 668), (967, 886)],
           [(598, 845), (787, 1208), (1043, 568), (1072, 1058)],
           [(598, 845), (787, 1208), (1043, 568), (1072, 1059)],
           [(598, 845), (787, 1208), (1043, 568), (1072, 1059)],
           [(666,666)],
           [(780, 1200), (1049, 941)]]

import numpy as np
            
# 
# centres = [[(782, 1221)],
#            [(967, 886)],
#            [(967, 885)],
#            [(967, 886)],
#            [(782, 1220)]]
 
# centres = [[(666, 666)],
#            [(967, 886)],
#             [(967, 885)],
#             [(967, 886)],
#             [(967, 886)],
#             [(838, 668), (967, 886)],
#             [(838, 668), (967, 886)],
#             [(838, 668), (967, 886)],
#             [(838, 668), (967, 886)],
#             [(838, 668), (967, 886)],
#             [(838, 668), (967, 886)],
#             [(838, 668), (967, 886)],
#             [(967, 886)],
#             [(666,666)],
#             [(967, 886)]]

# Pad the end of the list to enable removal
# of end elements
centres.append([(1,1)])
centres.append([(1,1)])

N = len(centres)
print "centre length before processing", len(centres)

# Find the (slice, index) of the elements which have no
# near neighbour, but are located further in the array
bad_pair = []
for slice in range(N - 2):
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
                for slice2 in range(slice + 3, N):
                    for index in range(len(centres[slice2])):
                        centre_further = centres[slice2][index]
                        #print "pair ", centre_further, centre
                        if np.allclose(np.asarray(centre), np.asarray(centre_further), 0, 15):
                            bad_pair.append((slice, centre_index))
                            
# Get unique pairs of the slices and indices to be
# removed
unique_pairs = []
unique_pairs.append(bad_pair[0])
for index in range(len(bad_pair)):
    if bad_pair[index] in unique_pairs:
        continue
    else:
        unique_pairs.append(bad_pair[index])

# Remove the indices slices and elements in them
# which are anomalous
print unique_pairs
centres = np.asarray(centres)
for i in centres:
    print i
    
for i in range(len(unique_pairs), -1):
    print 
    slice = i[0]
    index = i[1]
    del centres[slice][index]

# If a list was completely wiped out
# it will be empty, hence delete empty lists


bad_indices = []
for n in range(N):
    empt = centres[n]
    if not empt:
        bad_indices.append(n)
        
centres = np.delete(centres, bad_indices)

print "centre length after processing", len(centres)
for i in centres:
    print i
    
N = len(centres)

index_top = [] # indexes of tops of spheres
centres_top = [] # centres of tops of spheres
index_bot = [] # indexes of bottoms of spheres
centres_bot = [] # centres of bottoms of spheres
centres_picked = [] # to store the (x,y) centres of spheres


"""for centre in centres[0]:
    centres_picked.append(centre)
    index_top.append(0)
    centres_top.append(centre)"""
#     
# Process
for slice in range(0, N-2):
    # Pick centres
    for centre in centres[slice]:
        # If the centre is not in the two following slices (to prevent from bugs): bottom of sphere
        for centre_fol in centres[slice+1]:
            # if centres equal to within about 10 pixels
            if np.allclose(np.asarray(centre), np.asarray(centre_fol), 0, 15): 
                break
        else:
            for centre_foll in centres[slice+2]:
                if np.allclose(np.asarray(centre), np.asarray(centre_foll), 0, 15):
                    break
            # if centres are not the same within two neighbouring
            # slices then add them to centre bot
            else:
                index_bot.append(slice)
                centres_bot.append(centre)
        # If the centre has not been picked before: top of sphere
        # If the neighbouring slices have centres close together
        # then add them to centres picked and top arrays
        for centre_p in centres_picked:
            if np.allclose(np.asarray(centre), np.asarray(centre_p), 0, 15):
                break
        else:
            centres_picked.append(centre)
            index_top.append(slice)
            centres_top.append(centre)

for slice in range(N-2,N-1):
    # Pick centres
    for centre in centres[slice]:
        # If the centre is not in the two following slices (to prevent from bugs): bottom of sphere
        for centre_fol in centres[slice+1]:
            # if centres equal to within about 10 pixels
            if np.allclose(np.asarray(centre), np.asarray(centre_fol), 0, 15): 
                break
        else:
            index_bot.append(slice)
            centres_bot.append(centre)
        # If the centre has not been picked before: top of sphere
        # If the neighbouring slices have centres close together
        # then add them to centres picked and top arrays
        for centre_p in centres_picked:
            if np.allclose(np.asarray(centre), np.asarray(centre_p), 0, 15):
                break
        else:
            centres_picked.append(centre)
            index_top.append(slice)
            centres_top.append(centre)

for centre in centres[N-1]:
    index_bot.append(N-1)
    centres_bot.append(centre)
            
print len(index_bot)
print len(index_top)

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
print len(index_bot)
print len(index_top)

# The centres correspond to the index array
# only centres_top corresponds to the indices
# index_bot is scrambled so it needs to be sorted 

print centres_bot
print centres_top
print "remove centres which repeat only once"
1
1