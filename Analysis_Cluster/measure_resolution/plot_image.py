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

#     fig = pl.figure()
#     ax = fig.gca(projection='3d')
#     for slice in range(N):
#         length = len(centroids_sphere[slice])
#         try:
#             for i in range(length):
#                 print int(centroids_sphere[slice][i][0])
#                 print int(centroids_sphere[slice][i][1])
#                 ax.plot(int(centroids_sphere[slice][i][0]), int(centroids_sphere[slice][i][1]), slice*step+step)
#         except:
#             ax.plot(int(centroids_sphere[slice][0][0]), int(centroids_sphere[slice][0][1]), slice*step+step)
#      
#     pl.title('Sphere detection on real image')
#     pl.show()
    
    fig = pl.figure()
    ax = fig.gca(projection='3d')
    for slice in range(N):
        for i in range(len(perimeters[slice])):
            ax.plot(perimeters[slice][i][0] + bord_circles[slice][i][0], perimeters[slice][i][1] + bord_circles[slice][i][2], slice*step+step)
           
    pl.title('Sphere detection on real image')
    pl.show()