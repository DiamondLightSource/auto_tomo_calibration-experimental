
def not_within_range(element, list):
    for item in list:
        if np.allclose(element, item, 0, 20):
            return False
        
    return True

import numpy as np

#centres = np.load('/home/jjl36382/auto_tomo_calibration-experimental/modified_centroids.npy')
centres = np.load('/home/jjl36382/auto_tomo_calibration-experimental/Analysis_Cluster/modified_centroids.npy')
radius = np.load('/home/jjl36382/auto_tomo_calibration-experimental/Analysis_Cluster/radius.npy')

# radius = np.load('/home/jjl36382/auto_tomo_calibration-experimental/Analysis_Cluster/radius.npy')

N = len(centres)

print "centres at the beginning"
for i in range(N):
    print "I: ", i, " ", centres[i]
    
dict = {}
dict_for_averaging = {}
dict_radius = {}

# Takes an element from every slice and loops through the array to check if
# if the same centres exist within three slices
for slice_index in range(N - 4):
    for centr in centres[slice_index]:
        len_counter = 0
        end_loop = False
        list_for_averaging = []
        rad_for_averaging = []
        # For each centre in the slice go through
        # the whole array of slices and count
        # brute force...
        # set up a variable for length
        # go to the next slice
        # WITHIN A CERTAIN RANGE
        if not_within_range(centr, dict.keys()):
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
                    for index in range(len(centres[slice_next])):
                        element = centres[slice_next][index]
                        if np.allclose(np.asarray(centr), np.asarray(element), 0, 20):
                            found_one = True
                            len_counter += 1
                            list_for_averaging.append(element)
                            rad_for_averaging.append(radius[slice_next][index])
                            break
                    else:
                        for index1 in range(len(centres[slice_next + 1])):
                            element = centres[slice_next + 1][index1]
                            if np.allclose(np.asarray(centr), np.asarray(element), 0, 20):
                                found_one = True
                                len_counter += 1
                                list_for_averaging.append(element)
                                rad_for_averaging.append(radius[slice_next + 1][index1])
                                break
                        else:
                            for index2 in range(len(centres[slice_next + 2])):
                                element = centres[slice_next + 2][index2]
                                if np.allclose(np.asarray(centr), np.asarray(element), 0, 20):
                                    found_one = True
                                    len_counter += 1
                                    list_for_averaging.append(element)
                                    rad_for_averaging.append(radius[slice_next + 2][index2])
                                    break
                            else:
                                for index3 in range(len(centres[slice_next + 3])):
                                    element = centres[slice_next + 3][index3]
                                    if np.allclose(np.asarray(centr), np.asarray(element), 0, 20):
                                        found_one = True
                                        len_counter += 1
                                        list_for_averaging.append(element)
                                        rad_for_averaging.append(radius[slice_next + 3][index3])
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
                            dict_radius[centr] = rad_for_averaging
                            end_loop = True
                        else:
                            continue

# Check if the lengths are more than 2
for centre in dict.iterkeys():
    slice_start = dict[centre][0]
    slice_end = dict[centre][1]
    # end is inclusive so add 1
    length = slice_end - slice_start + 1
    
    # also take the median of all the centre values
    avg = np.median(dict_for_averaging[centre], axis=0)
    dict_for_averaging[centre] = tuple(np.array(avg, dtype='int64'))
    
    avg_rad = np.max(dict_radius[centre])
    dict_radius[centre] = np.array(avg_rad)
    if length < 3:
        del dict[centre]
        del dict_radius[centre]
        
# take the mean value of the centre together
# with its lengths and make one dict
centroids = {}
radii = {}
for key in dict.iterkeys():
    centroids[dict_for_averaging[key]] = dict[key]
    radii[dict_for_averaging[key]] = dict_radius[key]

print centroids
print dict_radius

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
radii_list = []
for key in centroids.iterkeys():
    x = key[0]
    y = key[1]
    z = centroids[key]
    r = radii[key]
    centres_list.append((x, y, z))
    radii_list.append(r)

nb_spheres = len(centres_list)
    
print centres_list
print radii_list
print nb_spheres

# print("Saving data")
# 
# f = open('/dls/tmp/jjl36382/results/centres.txt', 'w')
# for i in range(nb_spheres):
#     f.write(repr(centres_list[i]) + '\n')
# f.close()
# 
# f = open('/dls/tmp/jjl36382/results/centresX.txt', 'w')
# for i in range(nb_spheres):
#     f.write(repr(centres_list[i][0]) + '\n')
# f.close()
# 
# f = open('/dls/tmp/jjl36382/results/centresY.txt', 'w')
# for i in range(nb_spheres):
#     f.write(repr(centres_list[i][1]) + '\n')
# f.close()
# 
# f = open('/dls/tmp/jjl36382/results/centresZ.txt', 'w')
# for i in range(nb_spheres):
#     f.write(repr(centres_list[i][2]) + '\n')
# f.close()
# 
# # f = open('/dls/tmp/jjl36382/results/radii.txt', 'w')
# # for i in range(nb_spheres):
# #     f.write(repr(radii_slices[i]) + '\n')
# # f.close()
# 
# # f = open('/dls/tmp/jjl36382/results/radii_max.txt', 'w')
# # for i in range(nb_spheres):
# #     f.write(repr(max(radii_slices[i])) + '\n')
# # f.close()
# 
# f = open('/dls/tmp/jjl36382/results/radii_max.txt', 'w')
# for i in range(nb_spheres):
#     f.write(repr(int(radii_list[i])) + '\n')
# f.close()
# 
# f = open('/dls/tmp/jjl36382/results/nb_spheres.txt', 'w')
# f.write(repr(nb_spheres))
# f.close()
