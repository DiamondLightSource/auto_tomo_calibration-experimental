# This is to plot detected circles on images with different amounts of noise

from matplotlib.pyplot import axes, figure

def test_detector(np_image):
    
    import numpy as np
    import pylab as pl
    from skimage.util import img_as_ubyte
    
    pl.close('all')
    
    runfile('C:\\Users\\eqp83935\\workspace\\data\\src\\simple_circle_detector.py')
    runfile('C:\\Users\\eqp83935\\workspace\\data\\src\\filter.py')
    
    img_noise = []
    img_denoise = []
    C = []
    
    N = 10
    
    for i in range(N):
        img_noise.append(add_noise(np_image, round(0.1*i, 1)))
    
    for i in range(N):
        if i<2:
            img_denoise.append(img_noise[i])
        if i>=2:
            if np.min(img_noise[i])<0:
                img_noise[i] = img_noise[i] - np.min(img_noise[i])
             
            img_denoise.append(filter(img_noise[i]))
    
    for i in range(N):
        C.append(detect_circles(img_denoise[i]))
    
    fig = pl.figure(figsize=(22,8.5))
    
    for i in range(N):
        pl.subplot(2, 5, i+1)
        pl.imshow(C[i], aspect='auto')
        pl.tight_layout()
        pl.title('Noise = ' + repr(round(0.1*i, 1)))
        pl.axis('off')
    
    pl.show()
    
    return