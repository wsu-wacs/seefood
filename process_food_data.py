import os
import numpy as np

"""
Routines to fetch and process image data from disk

For CEG 4110 by: Derek Doran
contact: derek.doran@wright.edu
Aug 3 2017
"""
def getData(rootdir, percent_food_to_use = 1, check = False):
    """
    :param rootdir: Path to the Training directory from the ade20k dataset (the full data).
    :return: A list of list. It has the following:
    0 list: 'image_path', the file path of all the images
    1 list: 'image_label', the scene label of all the images
    """
    #list to store image path
    image_path = []
    #list to store image scene label
    image_label = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if subdir[subdir.rfind("/")+1:] == "food":
                if np.random.uniform() < percent_food_to_use:
                    image_label.append(1)
                    image_path.append(os.path.abspath(subdir)+"/"+file)
            elif subdir[subdir.rfind("/")+1:] == "not-food":
                image_label.append(0)
                image_path.append(os.path.abspath(subdir) + "/" + file)
            else:
                print "Error: image label not found. See filename and subdir below..."
                print file
                print subdir
                return


    return [image_path, image_label]