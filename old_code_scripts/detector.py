import os
from skimage import io

import circle_detector

def save_data(filename, data):
    import pickle
    print("Saving data")
    f = open(filename, 'w')
    pickle.dump(data, f)
    f.close()

if __name__ == '__main__' :
    import optparse
    usage = "%prog [options] input_file_template, output_file_template \n" + \
        "  input_file_template  = /location/of/file/filename%05i.tif \n" + \
        "  output_file_template = /location/of/output/filename%05i.tif"
    parser = optparse.OptionParser(usage=usage)

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
    result = circle_detector.detect_circles(image)

    # save image
    print("Saving image %s" % output_filename)
    save_data(output_filename, result)
