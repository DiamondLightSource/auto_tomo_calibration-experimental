import cPickle
import numpy as np
import optparse


def create_dir(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
def not_within_range(element, list, value):
    for item in list:
        if np.allclose(element, item, 0, value):
            return False
        
    return True


def save_data(filename, data):
    print("Saving data")
    f = open(filename, 'w')
    cPickle.dump(data, f)
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
    tol = 50
    
    """
    Loads all of the detected parameters of segmented circles
    in a slice and appends them to a list.
    """
    centres = []
    radius = []
    edges = []
    for i in range(start, stop, step):
        try:
            f = open(input_filename % i, 'r')
            data = cPickle.load(f)
            centres.append(data[0])
            radius.append(data[1])
            edges.append(data[2])
            f.close()
        except:
            print "MISSING FRAME %05i" % i
            centres.append([])
            radius.append([])
            edges.append([])

    N = len(centres)
    

#     import pylab as pl
#     from mpl_toolkits.mplot3d import Axes3D
#       
#     fig = pl.figure()
#     ax = fig.gca(projection='3d')
#     
#     for slice in range(N):
#         for i in range(len(edges[slice])):
#             ax.plot(edges[slice][i][0], edges[slice][i][1], slice*step+step, solid_joinstyle='bevel')
# 
#     pl.axis('off')
#     pl.savefig(results_folder + "/full_plot.png")
#     pl.close('all')
    

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

    dict = {}
    dict_for_averaging = {}
    dict_radius = {}
    dict_edge = {}
    
    # Takes an element from every slice and loops through the array to check if
    # if the same centres exist within three slices
    for slice_index in range(N - 8):
        for centr in centres[slice_index]:
            len_counter = 0
            end_loop = False
            list_for_averaging = []
            rad_for_averaging = []
            shell = []
            # For each centre in the slice go through
            # the whole array of slices and count
            # brute force...
            # set up a variable for length
            # go to the next slice
            if not_within_range(centr, dict.keys(), 50):
                for slice_next in range(slice_index + 1, N - 7):
                         
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
                                shell.append(edges[slice_next][index])
                                break
                        else:
                            for index1 in range(len(centres[slice_next + 1])):
                                element = centres[slice_next + 1][index1]
                                if np.allclose(np.asarray(centr), np.asarray(element), 0, tol):
                                    found_one = True
                                    len_counter += 2
                                    list_for_averaging.append(element)
                                    rad_for_averaging.append(radius[slice_next + 1][index1])
                                    shell.append(edges[slice_next + 1][index1])
                                    break
                            else:
                                for index2 in range(len(centres[slice_next + 2])):
                                    element = centres[slice_next + 2][index2]
                                    if np.allclose(np.asarray(centr), np.asarray(element), 0, tol):
                                        found_one = True
                                        len_counter += 3
                                        list_for_averaging.append(element)
                                        rad_for_averaging.append(radius[slice_next + 2][index2])
                                        shell.append(edges[slice_next + 2][index2])
                                        break
                                else:
                                    for index3 in range(len(centres[slice_next + 3])):
                                        element = centres[slice_next + 3][index3]
                                        if np.allclose(np.asarray(centr), np.asarray(element), 0, tol):
                                            found_one = True
                                            len_counter += 4
                                            list_for_averaging.append(element)
                                            rad_for_averaging.append(radius[slice_next + 3][index3])
                                            shell.append(edges[slice_next + 3][index3])
                                            break
                                    else:
                                        for index4 in range(len(centres[slice_next + 4])):
                                            element = centres[slice_next + 4][index4]
                                            if np.allclose(np.asarray(centr), np.asarray(element), 0, tol):
                                                found_one = True
                                                len_counter += 5
                                                list_for_averaging.append(element)
                                                rad_for_averaging.append(radius[slice_next + 4][index4])
                                                shell.append(edges[slice_next + 4][index4])
                                                break
                                        else:
                                            for index5 in range(len(centres[slice_next + 5])):
                                                element = centres[slice_next + 5][index5]
                                                if np.allclose(np.asarray(centr), np.asarray(element), 0, tol):
                                                    found_one = True
                                                    len_counter += 6
                                                    list_for_averaging.append(element)
                                                    rad_for_averaging.append(radius[slice_next + 5][index5])
                                                    shell.append(edges[slice_next + 5][index5])
                                                    break
                                                
                                                
                        # If the element was n ot found within 3 slices
                        # then it does not form a sphere
                        # hence found_one will be False
                        # and this part will execute meaning the end
                        # of the sphere
                        if not found_one:
                            # start and end index 
                            dict[centr] = (slice_index * step + start, len_counter * step + slice_index + start)
                            dict_for_averaging[centr] = list_for_averaging
                            dict_radius[centr] = rad_for_averaging
                            dict_edge[centr] = shell
                            end_loop = True
    
    index = []
    median_centres = {}
    mean_radius = {}
    for centre in dict.iterkeys():
        
        slice_start = dict[centre][0]
        slice_end = dict[centre][1]
        print slice_start
        print slice_end
        # end is inclusive so add 1
        length = (slice_end - slice_start + 1)
        if length > 20:
            # also take the median of all the centre values
            avg = np.median(dict_for_averaging[centre], axis=0)
            
            median_centres[centre] = tuple(np.array(avg))
            
            avg_rad = np.max(dict_radius[centre])
            mean_radius[centre] = np.array(avg_rad)
        else:
            index.append(centre)
            
    for centre in index:
        del dict[centre]
        del dict_radius[centre]
        del dict_for_averaging[centre]
        del dict_edge[centre]
    
    
    sorted_centroids = {}
    sorted_edges = {}
    # Re-sort the values based on the median centroid value
    for centre in dict.iterkeys():
        median_centroid = median_centres[centre]
        centre_list = dict_for_averaging[centre]
        edge_list = dict_edge[centre]
        
        temp_centroid = []
        temp_edge = []

        for i in range(len(centre_list)):
            centroid = centre_list[i]
            edge = edge_list[i]
            
            if np.allclose(median_centroid, centroid, 0, 2):
                temp_centroid.append(centroid)
                temp_edge.append(edge)
            else:
                temp_centroid.append([])
                temp_edge.append([])
        
        sorted_centroids[centre] = temp_centroid
        sorted_edges[centre] = temp_edge
    
    ################### FIT SPHERES #######################
    
    sphere_centres = []
    
    # Take circle perimeters and merge them into one array
    from sphere_fit import leastsq_sphere

    for centre in dict.iterkeys():
        
        X_coords = []
        Y_coords = []
        Z_coords = []
        
        mean_centre = np.mean(sorted_centroids[centre], axis=0)
        perimeter = sorted_edges[centre]
        slice_start = dict[centre][0]
        rad = mean_radius[centre]

        for i in range(0, len(perimeter), 1):
            
            if perimeter[i] != False and perimeter[i] != []:
                length = len(perimeter[i][0])
                
                X_coords.extend(perimeter[i][0])
                Y_coords.extend(perimeter[i][1])
                Z_coords = np.hstack([np.repeat([slice_start], length), Z_coords])
                slice_start += step
            else:
                slice_start += step
        
        print median_centres[centre]
        # Once the coordinates are obtained fit the spheres
        p1, rsq = leastsq_sphere(X_coords, Y_coords, Z_coords, rad, mean_centre)
        x1, y1, z1, r1 = p1
        
        ################################################
        
        sphere_centres.append([x1, y1, z1, r1])
        print "DATA USING PERIMETERS FROM HOUGH CIRCLES"
        print "SPHERE CENTRE AND RADIUS", x1, y1, z1, r1
        print "R SQUARED", rsq
#         N = len(perimeter)
#         fig = pl.figure()
#         ax = fig.gca(projection='3d')
#         for slice in range(N):
#             if perimeter[slice] != []:
#                 ax.plot(perimeter[slice][0], perimeter[slice][1], slice*step)
#             else:
#                 ax.plot([], [], slice)
#    
#         pl.title('Segmented sphere')
#         pl.savefig(results_folder + "/true{0}.png".format((round(centre[0],2), round(centre[1],2))))
#         pl.close('all')
        
    
    ############################## SAVE DATA ##########################
    centres_list = []
    radii_list = []

    for i in range(len(sphere_centres)):
        
        x,y,z,r = sphere_centres[i]
        if abs(start - z) < r or abs(stop - z) < r:
            continue
        else:
            centres_list.append((x,y,z))
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
    
    save_data(results_folder + '/centresX.npy', centres_list)
    f = open(results_folder + '/centresX.txt', 'w')
    for i in range(nb_spheres):
        f.write(repr(centres_list[i][0]) + '\n')
    f.close()
    
    save_data(results_folder + '/centresY.npy', centres_list)
    f = open(results_folder + '/centresY.txt', 'w')
    for i in range(nb_spheres):
        f.write(repr(centres_list[i][1]) + '\n')
    f.close()
    
    save_data(results_folder + '/centresZ.npy', centres_list)
    f = open(results_folder + '/centresZ.txt', 'w')
    for i in range(nb_spheres):
        f.write(repr(centres_list[i][2]) + '\n')
    f.close()
    
    save_data(results_folder + '/radii.npy', radii_list)
    f = open(results_folder + '/radii.txt', 'w')
    for i in range(nb_spheres):
        f.write(repr(radii_list[i]) + '\n')
    f.close()