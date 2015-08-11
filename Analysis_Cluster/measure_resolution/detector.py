import os
from skimage import io
import optparse
import circle_detector
import detector_watershed

def save_data(filename, data):
    import pickle
    print("Saving data")
    f = open(filename, 'w')
    pickle.dump(data, f)
    f.close()

if __name__ == '__main__' :
    parser = optparse.OptionParser()

    (options, args) = parser.parse_args()

    # get the number of the frame to process
    task_id = int(os.environ['SGE_TASK_ID']) - 1

    # make the filename
    input_filename = args[0] % task_id
    output_filename = args[1] % task_id

    # load image
    print("Loading image %s" % input_filename)
    image = io.imread(input_filename)

    # process image
    print("Processing data")
    #result = circle_detector.detect_circles(image)
    # TODO: TRY WATERSHED SLICING INSTEAD OF HOUGH
    folder = "/dls/tmp/jjl36382/resolution1/label/"
    result = detector_watershed.watershed_segmentation(image, 3, folder, task_id)
    
    # save image
    print("Saving image %s" % output_filename)
    save_data(output_filename, result)
