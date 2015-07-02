from PIL import Image
import numpy as np
import pylab as pl
from scipy import misc
from fpformat import extract
import get_radii as gr

"""
Input:
    Image containing circles
    
Outputs:
    areas - the areas containing the segmented circles (edges)
    bord - the coordinates of the borders
    areas_full - the areas containing the full segmented disks

This function locates circles within the image and segments them.
This function is then used for Hough transform to reduce computation time.  

The image is denoised with Chambolle filter and Sobel edges are obtained.
After thresholding a binary image is obtained and using the standard functions
the parameters of each segmented box can be obtained  
"""


def select_area_for_detector(np_image):
     
    import numpy as np
    import pylab as pl
     
    from skimage.filter import threshold_otsu, sobel
    from skimage.morphology import closing, square,dilation,label
    from skimage.measure import regionprops
    from skimage.restoration import denoise_tv_chambolle
    from skimage.segmentation import clear_border
     
    pl.close('all')
    
    """#image_filtered = denoise_tv_chambolle(np_image, weight=0.008)
    # get the sobel edges
    #edges = sobel(image_filtered)
    nbins = 10
    threshold = threshold_otsu(np_image, nbins)
    
    #print 'Threshold = ' + repr(threshold)
    #threshold = 1
    bw = closing(np_image >= threshold, square(3))
    
    
    # Remove artifacts connected to image border
    
    cleared = bw.copy()
    clear_border(cleared)
    pl.subplot(1,3,2)
    pl.imshow(cleared)
    
    # Label image regions
    
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    
    pl.subplot(1,3,3)
    pl.imshow(label_image)
    
    pl.show()"""
    
    """
    PROCESS THE IMAGE BY DENOISING, GETING THE EDGES AND THEN FINALY
    THRESHOLDING IT. THE LABELED IMAGE IS THEN USED IN ANALYSIS
    """
    image_filtered = denoise_tv_chambolle(np_image, weight=0.002)
    edges = sobel(image_filtered)
    nbins = 2
    threshold = threshold_otsu(edges, nbins)
    edges_bin = edges >= threshold
    label_image = label(edges_bin)
    
    """ pl.subplot(1,3,1)
    pl.imshow(image_filtered)
    
    pl.subplot(1,3,2)
    pl.imshow(edges_bin)
    
    pl.subplot(1,3,3)
    pl.imshow(label_image)
    pl.show()"""
    
    areas = []
    areas_full = []
    bord = []
    
    """
    REGIONPROPS FUNCTION USES THE LABELED IMAGE TO EXTRACT
    INFORMATION ABOUT THE CIRCLES
    """ 
    for region in regionprops(label_image, ['Area', 'BoundingBox', 'Label']):
         
        # Skip wrong regions
        index = np.where(label_image == region['Label'])
        if index[0].size==0 & index[1].size==0 :
            continue
         
        # Skip small images
        """if region['Area'] < 100:
            continue"""
         
        # Extract the coordinates of regions
        minr, minc, maxr, maxc = region['BoundingBox']
        margin = 2#len(np_image) / 100
        bord.append((minr - margin, maxr + margin, minc - margin, maxc + margin))
        areas.append(edges_bin[minr - margin:maxr + margin, minc - margin:maxc + margin].copy())
        areas_full.append(np_image[minr - margin:maxr + margin, minc - margin:maxc + margin].copy())
        
    return areas, areas_full, bord

"""
Input:
    Image containing circles
    
Outputs:
    areas - the areas containing the segmented circles
    circles - the X and Y coordinates of circle perimeter
    C - the centers of the circles relative to the boxes (areas)
       

This function uses the previous function to obtain the data about
segmented images. That data is then processed to remove too small/big
areas.

After that Hough transform is applied to obtain the edges (perimeters)
and centers of the segmented circles. The segmented circles are then drawn
on the full image with a red color.
"""


def detect_circles(np_image):
     
    import numpy as np
    import pylab as pl
     
    from skimage.transform import hough_circle
    from skimage.feature import peak_local_max
    from skimage.draw import circle_perimeter
    from skimage import color
    from skimage.filter import threshold_otsu,sobel
    from skimage.morphology import closing,square
     
    pl.close('all')
    
    areas, areas_full, bord = select_area_for_detector(np_image)

    print 'Length of areas: ' + repr(len(areas))
    
    
    """
    AUTOMATIC CALIBRATION IS REQUIRED IN THIS CODE TO DELETE AREAS
    
    DELETES VERY SMALL OR VERY BIG AREAS OR SIMILAR AREAS
    """
    size = max(np_image.shape[0] / 2, np_image.shape[1] / 2)
    
    index = []
    for i in range(0, len(areas)):
        
        # Jump too big or too small areas
        if areas[i].shape[0] >= size or areas[i].shape[1] >= size\
        or areas[i].shape[0] <= size/3 or areas[i].shape[1] <= size/3\
        or abs(areas[i].shape[0] - areas[i-1].shape[0]) <= 2:
            index.append(i)
            continue
    
    if index != []:
        areas[:] = [item for i,item in enumerate(areas) if i not in index]
        bord[:] = [item for i,item in enumerate(bord) if i not in index]
  
    print 'Length of areas after selection: ' + repr(len(areas))
    
    
    """
    HOUGH TRANSOFRM TO EXTRACT CIRCLES AND PUT THE PERIMETERS INTO AN ARRAY
    
    IT ALSO PROCESSES THE SEGMENTED AREAS
    """
    circles = []  # to get the outlines of the circles
    C = []  # to get the centres of the circles, in relation to the different areas
    rad = []
        
    for i in range(0, len(areas)):
        """
        if (areas[i].dtype=='float_' or areas[i].dtype=='float16' or\
        areas[i].dtype=='float32') and not(np.array_equal(np.absolute(areas[i])>1,\
        np.zeros(areas[i].shape, dtype=bool))):
            image_norm = areas[i]/np.linalg.norm(areas[i])
        else:
            image_norm = areas[i]
        edges = sobel(image_norm)
        edges_closed = closing(edges, square(2))
        nbins = 50
        threshold = threshold_otsu(edges, nbins)
        edges_bin = edges_closed >= threshold
        
        pl.subplot(1,2,1)
        pl.imshow(areas[i])
        pl.subplot(1,2,2)
        pl.imshow(edges_bin)
        pl.show()
        """

        min_rad = int(max(areas[i].shape[0], areas[i].shape[1]) / 4.0)
        max_rad = int(max(areas[i].shape[0], areas[i].shape[1]) / 2.0)
        step = 1
         
        hough_radii = np.arange(min_rad, max_rad, step, np.float64)
        hough_res = hough_circle(areas[i], hough_radii)
        
        centers = []
        accums = []
        radii = []
        
        # For each radius, extract one circle
        for radius, h in zip(hough_radii, hough_res):
            peaks = peak_local_max(h, num_peaks=2)
            centers.extend(peaks)
            accums.extend(h[peaks[:, 0], peaks[:, 1]])
            radii.extend([radius, radius])
        
        for idx in np.argsort(accums)[::-1][:1]:
            center_x, center_y = centers[idx]
            C.append((center_x, center_y))
            radius = radii[idx]
            cx, cy = circle_perimeter(center_y, center_x, np.int64(radius))
            circles.append((cy, cx))
    	
    	rad.append(radii)
    
    """
    PLOT THE IMAGE IN A PROPER FORMAT wITH THICKER LINES
    """
    np_image_norm = np_image.copy()
    np_image_norm = color.gray2rgb(np_image_norm)

    for i in range(len(areas)):
        np_image_norm[circles[i][0] + bord[i][0], circles[i][1] + bord[i][2]] = (220,20,20)
        # To thicken the line by a few pixels
        np_image_norm[circles[i][0]-1 + bord[i][0], circles[i][1]-1 + bord[i][2]] = (220,20,20)
        np_image_norm[circles[i][0]-1 + bord[i][0], circles[i][1] + bord[i][2]] = (220,20,20)
        np_image_norm[circles[i][0]-1 + bord[i][0], circles[i][1]+1 + bord[i][2]] = (220,20,20)
        np_image_norm[circles[i][0] + bord[i][0], circles[i][1]-1 + bord[i][2]] = (220,20,20)
        np_image_norm[circles[i][0] + bord[i][0], circles[i][1]+1 + bord[i][2]] = (220,20,20)
        np_image_norm[circles[i][0]+1 + bord[i][0], circles[i][1]-1 + bord[i][2]] = (220,20,20)
        np_image_norm[circles[i][0]+1 + bord[i][0], circles[i][1] + bord[i][2]] = (220,20,20)
        np_image_norm[circles[i][0]+1 + bord[i][0], circles[i][1]+1 + bord[i][2]] = (220,20,20)
        
    pl.imshow(np_image_norm)
    pl.title('Circle detection on real image using Hough transform\n- optimised with image labelling algorithm -', fontdict={'fontsize': 20,'verticalalignment': 'baseline','horizontalalignment': 'center'})
    pl.colorbar()
    #pl.show()
        
    """
    If the circle is an odd number of pixels wide, then that will displace the center by one pixel and give
    an uncertainty in the radius, producing the sine wave shape.
    
    THERE IS NO CLEAR WAY WHETHER TO SUBTRACT OR TO ADD THE HALF RADIUS
    """
    C_cp = C
    C = []
    
    for i in range(len(areas)):
        try:
            circle_widthY = bord[i][1] - bord[i][0]
            circle_widthX = bord[i][2] - bord[i][3]

        except IndexError:
            return 0
        
        if circle_widthX % 2 != 0 and circle_widthY % 2 != 0:
            C.append((C_cp[i][0] + 0.5, C_cp[i][1] + 0.5))
        elif circle_widthX % 2 != 0:
            C.append((C_cp[i][0] + 0.5, C_cp[i][1]))
        elif circle_widthY % 2 != 0:
            C.append((C_cp[i][0], C_cp[i][1] + 0.5))
        else:
            C.append((C_cp[i][0], C_cp[i][1]))
	
    return areas, circles, C, bord, rad

# analyse an image
#img = np.array(Image.open("noisy_circles.png").convert('L'))
# get all data about the images
#areas, circles, centres, bord, radii = detect_circles(img)


"""def add_noise(np_image, amount):
    import numpy as np
    import pylab as pl
    from scipy import misc
    
    noise = np.random.randn(np_image.shape[0],np_image.shape[1])
    norm_noise = noise/np.max(noise)
    np_image = np_image + norm_noise*np.max(np_image)*amount
    pl.imshow(np_image, cmap=pl.cm.Greys)
    pl.show()
    misc.imsave("noisy_circles.png",np_image)
    return np_image"""

#noisy = add_noise(img, 0.9)


radii = []
img = np.array(Image.open("noisy_circles.png").convert('L'))
rad = gr.plot_radii(img, (64,65.5))
radii.append(rad)

print np.mean(radii)