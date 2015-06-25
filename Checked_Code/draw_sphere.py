def display(centres, radii):
    
    import numpy as np
    import pylab as pl
    from mpl_toolkits.mplot3d import Axes3D
    
    pl.close('all')
    
    '''DON'T FORGET TO CHANGE N'''
    
    N = 200
    
    fig = pl.figure()
    # chooses a 3D axes projection
    image_3d = fig.add_subplot(111, projection='3d')
    
    # linspace(start, stop, number of intervals)
    # arrays containing angles theta and phi
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    # x = r cosT sinP
    # y = r sinT sinP
    # z = r cosP
    # multiplies the outer product matrix by the radius
    # to define the sphere boundaries and shifts 
    for i in range(len(radii)):
        x = radii[i] * np.outer(np.cos(u), np.sin(v)) + centres[i][0]
        y = radii[i] * np.outer(np.sin(u), np.sin(v)) + centres[i][1]
        z = radii[i] * np.outer(np.ones(np.size(u)), np.cos(v)) + centres[i][2]
        image_3d.plot_surface(x, y, z, rstride=4, cstride=4, color='r')
    
    image_3d.set_xlim(0,N)
    image_3d.set_ylim(0,N)
    image_3d.set_zlim(0,N)
    pl.title('Test image - 3D')
    
    pl.show()
    
    return image_3d

"""
generate test spheres
"""
centre = [(50,50,50), (80,100,130)]
radius = [30,60]
value = 10, 15
display(centre,radius)


