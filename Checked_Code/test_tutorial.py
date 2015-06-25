import circle_detector as cd
import get_radii as gr

from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises

from PIL import Image
import numpy as np
import numpy.testing as npt

class TestClass:
	
	# analyse an image
	img = np.array(Image.open("./Test_Images/real.png").convert('L'))
	# get all data about the images
	areas, circles, centres, bord, radii = cd.detect_circles(img)

	# get the cropped circles out of the image
	crops = []
	for i in xrange(0,len(areas)):
		crop = np.zeros(areas[i].shape)	
		crop[circles[i][0],circles[i][1]] = 255
		crops.append(crop)
		
	# test the radii
	radii = []
	radii_flat = []
	for i in xrange(0,len(areas)):
		rad, rad_flat = gr.remove_large_sine(crops[i], centres[i])
		radii.append(rad)
		radii_flat.append(rad_flat)
	
	def setUp(self):
		print "setting up a test..."

	def tearDown(self):
		print "tearing down after a test..."

	def test_exists(self):
		assert (len(self.areas) != 0)

	def test_radii(self):
		for i in xrange(len(self.areas)):
			npt.assert_almost_equal(self.radii[i],self.radii_flat[i])
			print self.radii[i], self.radii_flat[i]
