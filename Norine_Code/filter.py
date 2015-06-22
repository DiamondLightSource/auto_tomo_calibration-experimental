def filter(np_image, size):
    
    from skimage.filter import denoise_tv_chambolle, denoise_bilateral
    from scipy.ndimage import median_filter
    import pylab as pl
    
    #pl.close('all')
    
    imgm = median_filter(np_image, size)
    
    #pl.imshow(imgm, cmap=cm.gray)
    #pl.show()
    
    return imgm