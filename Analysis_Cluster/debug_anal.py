import numpy as np

centres = np.load('/home/jjl36382/auto_tomo_calibration-experimental/modified_centroids.npy')

# radius = np.load('/home/jjl36382/auto_tomo_calibration-experimental/Analysis_Cluster/radius.npy')

N = len(centres)

print "centres at the beginning"
for i in range(N):
    print "I: ", i, " ", centres[i]
    
# Find the (slice, index) of the elements which have no
# near neighbours and remove them
# However, they can be neighbours but only across two slices
# then this might be not too significant to be taken into account
# bad_pair = []
# for slice in range(0, N - 4):
#     for centre_index in range(len(centres[slice])):
#         centre = centres[slice][centre_index]
#         for centre_next in centres[slice + 1]:
#             # if the centres are within the tolerance
#             # and they are neighbours then stop
#             if np.allclose(np.asarray(centre), np.asarray(centre_next), 0, 20):
#                 break
#         else:
#             for centre_next2 in centres[slice + 2]:
#                 if np.allclose(np.asarray(centre), np.asarray(centre_next2), 0, 20):
#                     break
#             else:
#                 for centre_next3 in centres[slice + 3]:
#                     if np.allclose(np.asarray(centre), np.asarray(centre_next3), 0, 20):
#                         break
#                 else:
#                     bad_pair.append((slice, centre_index))
#                         
#  
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


dict = {}
dict_for_averaging = {}
radius = {}

# Takes an element from every slice and loops through the array to check if
# if the same centres exist within three slices
for slice_index in range(N - 4):
    for centr in centres[slice_index]:
        len_counter = 0
        end_loop = False
        list_for_averaging = []
        radius = []
        # For each centre in the slice go through
        # the whole array of slices and count
        # brute force...
        # set up a variable for length
        # go to the next slice
        if centr not in dict.keys():
            for slice_next in range(slice_index + 1, N - 3):
                
#                 Check if the array has not ended
#                 Since padding was used just skip this
#                 if slice_index == N - 5:
#                     # start and end index 
#                     dict[centr] = (slice_index, len_counter + slice_index)
#                     dict_for_averaging[centr] = list_for_averaging
#                     end_loop = True
                        
                # If one element was found in the slice then this is True
                # otherwise make it False
                found_one = False
                
                # this allows to loop through till the centre has neighbours
                # and reaches the end of all the slices
                # MIGHT NOT BE NECESSARY
                if not end_loop:
                    
                    # check if it is similar to one element
                    # in the next three slices
                    # if it is then increase the counter and
                    # say to the code that the element was found
                    # also append to the list to take the
                    # average of the centre
                    for element in centres[slice_next]:                    
                        if np.allclose(np.asarray(centr), np.asarray(element), 0, 20):
                            found_one = True
                            len_counter += 1
                            list_for_averaging.append(element)
                            break
                    else:
                        for element in centres[slice_next + 1]:                        
                            if np.allclose(np.asarray(centr), np.asarray(element), 0, 20):
                                found_one = True
                                len_counter += 1
                                list_for_averaging.append(element)
                                break
                        else:
                            for element in centres[slice_next + 2]:
                                if np.allclose(np.asarray(centr), np.asarray(element), 0, 20):
                                    found_one = True
                                    len_counter += 1
                                    list_for_averaging.append(element)
                                    break
                            else:
                                for element in centres[slice_next + 3]:
                                    if np.allclose(np.asarray(centr), np.asarray(element), 0, 20):
                                        found_one = True
                                        len_counter += 1
                                        list_for_averaging.append(element)
                                        break
                    # If the element was n ot found within 3 slices
                    # then it does not form a sphere
                    # hence found_one will be False
                    # and this part will execute meaning the end
                    # of the sphere
                    if not found_one:
                        if len_counter > 2:
                            # start and end index 
                            dict[centr] = (slice_index, len_counter + slice_index)
                            dict_for_averaging[centr] = list_for_averaging
                            end_loop = True
                        else:
                            continue
                        
# Remove the paddded lists
# centres = np.delete(centres, -1)
# centres = np.delete(centres, -1)
# centres = np.delete(centres, -1)
# centres = np.delete(centres, -1)
# centres = np.delete(centres, -1)
# centres = np.delete(centres, -1)


# Check if the lengths are more than 2
for centre in dict.iterkeys():
    slice_start = dict[centre][0]
    slice_end = dict[centre][1]
    # end is inclusive so add 1
    length = slice_end - slice_start + 1
    
    # also take the median of all the centre values
    avg = np.median(dict_for_averaging[centre], axis=0)
    dict_for_averaging[centre] = tuple(np.array(avg, dtype='int64'))
    print avg
    if length < 3:
        del dict[centre]
        
# take the mean value of the centre together
# with its lengths and make one dict
centroids = {}
for key in dict.iterkeys():
    centroids[dict_for_averaging[key]] = dict[key]

print centroids

# find the z position of the centres
# not sure about the plus 10 - must check
for key in centroids.iterkeys():
    slice_start = centroids[key][0]
    slice_end = centroids[key][1]
    z = (slice_end + slice_start) / 2.0
    centroids[key] = int(z  * 10 + 10)

print centroids 
# make a list with x,y,z coordinates
centres_list = []
for key in centroids.iterkeys():
    x = key[0]
    y = key[1]
    z = centroids[key]
    centres_list.append((x, y, z))
    
print centres_list
nb_spheres = len(centres_list)

print("Saving data")

f = open('/dls/tmp/jjl36382/results/centres.txt', 'w')
for i in range(nb_spheres):
    f.write(repr(centres_list[i]) + '\n')
f.close()

f = open('/dls/tmp/jjl36382/results/centresX.txt', 'w')
for i in range(nb_spheres):
    f.write(repr(centres_list[i][0]) + '\n')
f.close()

f = open('/dls/tmp/jjl36382/results/centresY.txt', 'w')
for i in range(nb_spheres):
    f.write(repr(centres_list[i][1]) + '\n')
f.close()

f = open('/dls/tmp/jjl36382/results/centresZ.txt', 'w')
for i in range(nb_spheres):
    f.write(repr(centres_list[i][2]) + '\n')
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
