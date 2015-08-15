import pickle
import numpy as np
import optparse

# TODO: THE TOLERANCE VALUES ARE NOT ROBUST

def create_dir(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def not_within_range(element, list):
    for item in list:
        if np.allclose(element, item, 0, 20):
            return False
        
    return True


def save_data(filename, data):
    print("Saving data")
    f = open(filename, 'w')
    pickle.dump(data, f)
    f.close()
    

tol = 40

if __name__ == '__main__' :
    """
    Loads all of the detected parameters of segmented circles
    in a slice and appends them to a list.
    """
    parser = optparse.OptionParser()
      
    (options, args) = parser.parse_args()

    start = int(args[0]) - 1
    stop = int(args[1]) - 1
    step = int(args[2])
    input_filename = args[3]
    results_folder = args[4]

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
        centroids_sphere.append(data[i][0])
        radii_circles.append(data[i][1])

    
    N = len(centroids_sphere)
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
            cx = centroids_sphere[slice][i][0]
            cy = centroids_sphere[slice][i][1]
            rad = radii_circles[slice][i]
            # IF THERE ARE SAME CENTRES IN THE SAME SLICE THEN
            # WE WILL GET ERRORS - REMOVE THEM
            cxcy[:] = [item for item in cxcy if not np.allclose(np.asarray((cx,cy)), np.asarray(item), 0, tol)]
            r[:] = [rad for item in cxcy if not np.allclose(np.asarray((cx,cy)), np.asarray(item), 0, tol)]
            # MODIFY THE RADIUS ACCORDINGLY
            r.append((round(rad,2)))
            cxcy.append((round(cx,2),round(cy,2)))
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
    centres.append([(1,1)])
    centres.append([(1,1)])
    radius.append([(666)])
    radius.append([(666)])
    radius.append([(666)])
    radius.append([(666)])
    radius.append([(666)])
    radius.append([(666)])
    radius.append([(666)])
    radius.append([(666)])
    N = len(centres)
    #--------------------------------------------------------
    
    N = len(centres)
    
    
    dict = {}
    dict_for_averaging = {}
    dict_radius = {}
    for i in centres:
        if i:
            print i
    # Takes an element from every slice and loops through the array to check if
    # if the same centres exist within three slices
    for slice_index in range(N - 6):
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
            if not_within_range(centr, dict.keys()):
                for slice_next in range(slice_index + 1, N - 5):
                    
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
                            if np.allclose(np.asarray(centr), np.asarray(element), 0, tol):
                                found_one = True
                                len_counter += 1
                                list_for_averaging.append(element)
                                rad_for_averaging.append(radius[slice_next][index])
                                break
                        else:
                            for index1 in range(len(centres[slice_next + 1])):
                                element = centres[slice_next + 1][index1]
                                if np.allclose(np.asarray(centr), np.asarray(element), 0, tol):
                                    found_one = True
                                    len_counter += 1
                                    list_for_averaging.append(element)
                                    rad_for_averaging.append(radius[slice_next + 1][index1])
                                    break
                            else:
                                for index2 in range(len(centres[slice_next + 2])):
                                    element = centres[slice_next + 2][index2]
                                    if np.allclose(np.asarray(centr), np.asarray(element), 0, tol):
                                        found_one = True
                                        len_counter += 1
                                        list_for_averaging.append(element)
                                        rad_for_averaging.append(radius[slice_next + 2][index2])
                                        break
                                else:
                                    for index3 in range(len(centres[slice_next + 3])):
                                        element = centres[slice_next + 3][index3]
                                        if np.allclose(np.asarray(centr), np.asarray(element), 0, tol):
                                            found_one = True
                                            len_counter += 1
                                            list_for_averaging.append(element)
                                            rad_for_averaging.append(radius[slice_next + 3][index3])
                                            break
                                    else:
                                        for index4 in range(len(centres[slice_next + 4])):
                                            element = centres[slice_next + 4][index4]
                                            if np.allclose(np.asarray(centr), np.asarray(element), 0, tol):
                                                found_one = True
                                                len_counter += 1
                                                list_for_averaging.append(element)
                                                rad_for_averaging.append(radius[slice_next + 4][index4])
                                                break
                                        else:
                                            for index5 in range(len(centres[slice_next + 5])):
                                                element = centres[slice_next + 5][index5]
                                                if np.allclose(np.asarray(centr), np.asarray(element), 0, tol):
                                                    found_one = True
                                                    len_counter += 1
                                                    list_for_averaging.append(element)
                                                    rad_for_averaging.append(radius[slice_next + 5][index5])
                                                    break
                                                
                                                
                        # If the element was n ot found within 3 slices
                        # then it does not form a sphere
                        # hence found_one will be False
                        # and this part will execute meaning the end
                        # of the sphere
                        if not found_one:
                            # start and end index 
                            dict[centr] = (slice_index, len_counter + slice_index)
                            dict_for_averaging[centr] = list_for_averaging
                            dict_radius[centr] = rad_for_averaging
                            end_loop = True
    
    index = []
    for centre in dict.iterkeys():
        
        slice_start = dict[centre][0]
        slice_end = dict[centre][1]
        # end is inclusive so add 1
        length = slice_end - slice_start + 1
        print length
        if length > 100:
            # also take the median of all the centre values
            avg = np.mean(dict_for_averaging[centre], axis=0)
            dict_for_averaging[centre] = tuple(np.array(avg))
            
            avg_rad = np.max(dict_radius[centre])
            dict_radius[centre] = np.array(avg_rad)
        else:
            dict_for_averaging[centre] = False
            dict_radius[centre] = False
            
            index.append(centre)
            
    for centre in index:
        del dict[centre]
        del dict_radius[centre]
        del dict_for_averaging[centre]
        
    # take the mean value of the centre together
    # with its lengths and make one dict
    centroids = {}
    radii = {}
    for key in dict.iterkeys():
        centroids[dict_for_averaging[key]] = dict[key]
        radii[dict_for_averaging[key]] = dict_radius[key]
    
    
    # find the z position of the centres
    # not sure about the plus 10 - must check
    for key in centroids.iterkeys():
        slice_start = centroids[key][0]
        slice_end = centroids[key][1]
        z = (slice_end + slice_start) / 2.0
        centroids[key] = z
    
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
        
    print "centres", centres_list
    print "radii", radii_list
    nb_spheres = len(centres_list)
    print "nb_spheres", nb_spheres
    

    save_data(results_folder + '/nb_spheres.npy', nb_spheres)
    f = open(results_folder + '/nb_spheres.txt', 'w')
    f.write(repr(nb_spheres))
    f.close()
    
    save_data(results_folder + '/centres.npy', centres_list)
    f = open(results_folder + '/centres.txt', 'w')
    for i in range(nb_spheres):
        f.write(repr(centres_list[i]) + '\n')
    f.close()
     
    
    max_radii = []
    for i in range(nb_spheres):
        max_radii.append(np.max(radii_list[i]))


    save_data(results_folder + '/radii.npy', max_radii)
    f = open(results_folder + '/radii.txt', 'w')
    f.write(repr(max_radii) + '\n')
    f.close()