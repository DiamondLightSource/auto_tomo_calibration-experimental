import pickle
import pylab as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import optparse


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
        bord_circles.append(data[i][2])
        centroids_sphere.append(data[i][0])
        radii_circles.append(data[i][3])
        perimeters.append(data[i][4])
        
    N = len(centroids_sphere)

    # fig = pl.figure()
    # ax = fig.gca(projection='3d')
    # for slice in range(N):
    #     for i in range(len(centroids_sphere[slice])):
    #         ax.plot(centroids_sphere[slice][i][0], centroids_sphere[slice][i][1], slice)
    # 
    # pl.title('Sphere detection on real image')
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
            cx = centroids_sphere[slice][i][0]
            cy = centroids_sphere[slice][i][1]
            rad = radii_circles[slice][i]
            # IF THERE ARE SAME CENTRES IN THE SAME SLICE THEN
            # WE WILL GET ERRORS - REMOVE THEM
            cxcy[:] = [item for item in cxcy if not np.allclose(np.asarray((cx,cy)), np.asarray(item), 0, 15)]
            r[:] = [rad for item in cxcy if not np.allclose(np.asarray((cx,cy)), np.asarray(item), 0, 15)]
            # MODIFY THE RADIUS ACCORDINGLY
            r.append((rad))
            cxcy.append((cx,cy))
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
    
    N = len(centres)


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
                            if np.allclose(np.asarray(centr), np.asarray(element), 0, 15):
                                found_one = True
                                len_counter += 1
                                list_for_averaging.append(element)
                                rad_for_averaging.append(radius[slice_next][index])
                                break
                        else:
                            for index1 in range(len(centres[slice_next + 1])):
                                element = centres[slice_next + 1][index1]
                                if np.allclose(np.asarray(centr), np.asarray(element), 0, 15):
                                    found_one = True
                                    len_counter += 1
                                    list_for_averaging.append(element)
                                    rad_for_averaging.append(radius[slice_next + 1][index1])
                                    break
                            else:
                                for index2 in range(len(centres[slice_next + 2])):
                                    element = centres[slice_next + 2][index2]
                                    if np.allclose(np.asarray(centr), np.asarray(element), 0, 15):
                                        found_one = True
                                        len_counter += 1
                                        list_for_averaging.append(element)
                                        rad_for_averaging.append(radius[slice_next + 2][index2])
                                        break
                                else:
                                    for index3 in range(len(centres[slice_next + 3])):
                                        element = centres[slice_next + 3][index3]
                                        if np.allclose(np.asarray(centr), np.asarray(element), 0, 15):
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
    
    
    # find the z position of the centres
    # not sure about the plus 10 - must check
    for key in centroids.iterkeys():
        slice_start = centroids[key][0]
        slice_end = centroids[key][1]
        z = (slice_end + slice_start) / 2.0
        centroids[key] = int(z  * step) + start
    
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
    
        # Store data
    f = open(results_folder + '/nb_spheres.txt', 'w')
    f.write(repr(nb_spheres))
    f.close()
 
    f = open(results_folder + '/centres.txt', 'w')
    for i in range(nb_spheres):
        f.write(repr(centres_list[i]) + '\n')
    f.close()
     
    f = open(results_folder + '/radii.txt', 'w')
    for i in range(nb_spheres):
        f.write(repr(np.max(radii_list[i])) + '\n')
    f.close()