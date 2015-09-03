import os
from skimage import io
import optparse
import circle_detector


def save_data(filename, data):
    import cPickle
    print("Saving data")
    f = open(filename, 'w')
    cPickle.dump(data, f)
    f.close()


if __name__ == '__main__' :
    """
    It gets the data from the shell script
    and sends image files to be processed at
    circle_detector.py
    
    Then it gets back the data and saves it.
    """
    parser = optparse.OptionParser()

    (options, args) = parser.parse_args()

    # get the number of the frame to process
    task_id = int(os.environ['SGE_TASK_ID']) - 1

    # make the filename
    input_filename = args[0] % task_id
    output_filename = args[1] % task_id
    label_filename = args[2]
    
    # load image
    print("Loading image %s" % input_filename)
    image = io.imread(input_filename)

    # process image
    print("Processing data")
    result = circle_detector.watershed_segmentation(image, label_filename, task_id)
    
    # save image
    print("Saving image %s" % output_filename)
    save_data(output_filename, result)
