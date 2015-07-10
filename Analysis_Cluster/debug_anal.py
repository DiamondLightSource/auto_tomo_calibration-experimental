import numpy as np

centres = np.load('/home/tomas/Documents/Diamond/modified_centroids.npy')

# radius = np.load('/home/jjl36382/auto_tomo_calibration-experimental/Analysis_Cluster/radius.npy')

N = len(centres)

print "centres at the beginning"
for i in range(N):
    print "I: ", i, " ", centres[i]
    
# Find the (slice, index) of the elements which have no
# near neighbours and remove them
# However, they can be neighbours but only across two slices
# then this might be not too significant to be taken into account
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
                    bad_pair.append((slice, centre_index))
                        
#-----------------------------------------------------------

# Get unique pairs of the slices and indices to be removed
# go from top to bottom to preserve indices
# if bad_pair:
#     unique_pairs = []
#     unique_pairs.append(bad_pair[0])
#     for index in range(1, len(bad_pair)):
#         if bad_pair[index] in unique_pairs:
#             continue
#         else:
#             unique_pairs.append(bad_pair[index])
# 
#     # Remove the indices slices and elements in them
#     # which are anomalous
#     centres = np.asarray(centres)
#     sorted_by_slice = sorted(unique_pairs, key=lambda tup: tup[0])
#     rev_unique = list(reversed(sorted_by_slice))
#     
#     # print "unique slices for deletion and reversed"
#     # print rev_unique
#     for i in rev_unique:
#         slice = i[0]
#         index = i[1]
#         centres[slice][index] = []
# #         radius[slice][index] = []
#         
#     # If a list was completely wiped out
#     # it will be empty, hence delete empty lists
#     bad_indices = []
#     N = len(centres)
#     for n in range(N):
#         for element in centres[n]:
#             if not element:
#                 bad_indices.append(n)
#             
#     centres = np.delete(centres, bad_indices)
#     radius = np.delete(radius, bad_indices)

# SORT THE SLICES BECAUSE THEY ARE ALL SCRAMBLED
# DIDN'T HELP
new_centre = []
for i in range(len(centres)):
    sorted_by_slice = sorted(centres[i], key=lambda tup: tup[0])
    new_centre.append(sorted_by_slice)

centres = new_centre
for i in range(N):
    print "I: ", i, " ", centres[i]
    
#-------------------------------------------------------------

dict = {}
dict_for_averaging = {}
N = len(centres)
for slice_index in range(N - 1):
    for centr in centres[slice_index]:
        len_counter = 0
        end_loop = False
        list_for_averaging = []
        # For each centre in the slice go through
        # the whole array of slices and count
        # brute force...
        # set up a variable for length
        # go to the next slice
        if centr not in dict.keys():
            for slice_next in range(slice_index + 1, N - 2):
                
                # If one element was found in the slice
                # make it true
                found_one = False
                
                # if the slice still has neighbours
                if not end_loop:
                    
                    # check if it is similar to one element
                    # in the next slice and increase its length
                    for element in centres[slice_next]:                    
                        if np.allclose(np.asarray(centr), np.asarray(element), 0, 20):
                            found_one = True
                            len_counter += 1
                            list_for_averaging.append(element)
                            break
                            if slice_index == N - 1:
                                if len_counter > 2:
                                            # start and end index 
                                            dict[centr] = (slice_index, len_counter + slice_index)
                                            dict_for_averaging[centr] = list_for_averaging
                                            end_loop = True
                    else:
                        for element2 in centres[slice_next + 1]:                        
                            if np.allclose(np.asarray(centr), np.asarray(element2), 0, 20):
                                found_one = True
                                len_counter += 1
                                list_for_averaging.append(element)
                                break
                        else:
                            for element3 in centres[slice_next + 2]:
                                if np.allclose(np.asarray(centr), np.asarray(element3), 0, 20):
                                    found_one = True
                                    len_counter += 1
                                    list_for_averaging.append(element)
                                    break
                            else:             
                                if not found_one:
                                    if len_counter > 2:
                                        # start and end index 
                                        dict[centr] = (slice_index, len_counter + slice_index)
                                        dict_for_averaging[centr] = list_for_averaging
                                        end_loop = True
                                    else:
                                        continue

N = len(centres)
print dict
print dict_for_averaging
# index_top = [] # indexes of tops of spheres
# centres_top = [] # centres of tops of spheres
# index_bot = [] # indexes of bottoms of spheres
# centres_bot = [] # centres of bottoms of spheres
# centres_picked = [] # to store the (x,y) centres of spheres
# 
# # If a slice does not have neighbours within 4 slices
# # the it is a bug
# for slice in range(0, N-4):
#     # Pick centres
#     for centre in centres[slice]:
#         # If the centre is not in the two following slices (to prevent from bugs): bottom of sphere
#         for centre_1 in centres[slice+1]:
#             # if centres equal to within about 10 pixels
#             if np.allclose(np.asarray(centre), np.asarray(centre_1), 0, 20): 
#                 break
#         else:
#             for centre_2 in centres[slice+2]:
#                 if np.allclose(np.asarray(centre), np.asarray(centre_2), 0, 20):
#                     break
#             else:
#                 for centre_3 in centres[slice+3]:
#                     if np.allclose(np.asarray(centre), np.asarray(centre_3), 0, 20):
#                         break
#                 # if centres are not the same within four neighbouring
#                 # slices then add them to centre bot
#                 else:
#                     index_bot.append(slice)
#                     centres_bot.append(centre)
#         
#         # If the centre has not been picked before: top of sphere
#         # If the neighbouring slices have centres close together
#         # then add them to centres picked and top arrays
#         for centre_p in centres_picked:
#             if np.allclose(np.asarray(centre), np.asarray(centre_p), 0, 20):
#                 break
#         else:
#             centres_picked.append(centre)
#             index_top.append(slice)
#             centres_top.append(centre)
# #             radius_top.append(radius[slice])
# 
# # FIX IF THE CENTRES STARTED ACCUMULATING NEAR THE END
# # AND THEN THE SLICES ENDED
# for centre in centres[N-1]:
#     index_bot.append(N-1)
#     centres_bot.append(centre)
# #     radius_bot.append(radius[N-1])
# 
# # Remove bugs = wrong centres that were detected only once
# index_top_del = []
# index_bot_del = []
# for i_top in range(len(index_top)):
#     if (index_top[i_top] in index_bot) and (centres_top[i_top] in centres_bot): # bugs are the only centres that appear once in both lists of edges
#         print index_top[i_top]
#         print centres_top[i_top]
#         i_bot = centres_bot.index((centres_top[i_top]))
#         index_top_del.append(i_top)
#         index_bot_del.append(i_bot)
#   
#           
# index_top[:] = [item for i,item in enumerate(index_top) if i not in index_top_del]
# centres_top[:] = [item for i,item in enumerate(centres_top) if i not in index_top_del]
# index_bot[:] = [item for i,item in enumerate(index_bot) if i not in index_bot_del]
# centres_bot[:] = [item for i,item in enumerate(centres_bot) if i not in index_bot_del]
# 
# 
# # remove the padded lists
# del centres_bot[-1]
# del centres_top[-1]
# del index_top[-1] 
# del index_bot[-1]
centres = np.delete(centres, -1)
centres = np.delete(centres, -1)
centres = np.delete(centres, -1)
centres = np.delete(centres, -1)
centres = np.delete(centres, -1)
centres = np.delete(centres, -1)

# print len(index_bot)
# print len(index_top)
# print len(centres_top)
# print len(centres_bot)
print "after removing distant centres"

# N = len(centres)
# for i in range(N):
#     print "I: ", i, " ", centres[i]
                                 
print "remove centres which repeat only once"
1
1