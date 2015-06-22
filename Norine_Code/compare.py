def compare(image1, image2):
    
    import pylab as pl
    
    pl.close('all')
    
    #for i in range(10):
    pl.subplot(1,2,1)
    pl.title('Image')
    pl.imshow(image1, cmap=pl.cm.gray)
    
    pl.subplot(1,2,2)
    pl.title('Denoised image')
    pl.imshow(image2, cmap=pl.cm.gray)
    
    pl.show()
    
    return