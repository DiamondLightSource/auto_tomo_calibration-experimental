def draw_sphere(np_image, centre, radius, value):
    
    import numpy as np
    
    N = 100
    
    Xc = centre[0]
    Yc = centre[1]
    Zc = centre[2]
    
    Y, X, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
    mask = (((X - Xc)**2 + (Y - Yc)**2 + (Z - Zc)**2) < radius**2)
    
    np_image[mask] = value
    
    for i in range(N):
        dnp.plot.image(image[:,:,i])
    
    return

def add_noise(np_image, amount):
    import numpy as np
    noise = np.random.randn(np_image.shape[0],np_image.shape[1])
    norm_noise = noise/np.max(noise)
    np_image = np_image + norm_noise*np.max(np_image)*amount
    dnp.plot.image(np_image)
    return np_image

def display(centres, radii):
    
    import numpy as np
    import pylab as pl
    from mpl_toolkits.mplot3d import Axes3D
    
    pl.close('all')
    
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    for i in range(len(radii)):
        x = radii[i] * np.outer(np.cos(u), np.sin(v)) + centres[i][0]
        y = radii[i] * np.outer(np.sin(u), np.sin(v)) + centres[i][1]
        z = radii[i] * np.outer(np.ones(np.size(u)), np.cos(v)) + centres[i][2]
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color='r')
    
    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.set_zlim(0,100)
    
    pl.show()
    
    return