import os
from subprocess import call

# Code in separate project that generates this color class
def get_color_from_cclass(cclass):
    # return int(4 * color[0] + 2 * color[1] + color[2])
    r = int(cclass / 4)
    g = int(cclass / 2 % 2)
    b = int(cclass % 2)
    return [r,g,b]

# Use this to create the lmdb from a listing file
# Listing file must have format "<path/to/file> <class>" on each line for each file
# note that the above is relative to the built caffe binary
def convert_to_lmdb(listing_file):
    call_array = ["../net_sources/convert_imageset", "../images/", "../net_sources/" + listing_file, "../net_sources/" + listing_file + "_lmdb"]
    call(call_array)

def create_listfile(image_dir):
    files = os.listdir("../images/" + image_dir)
    with open("../net_sources/images.txt", 'w') as out:
        for file in files:
            string = "{0}{1} {2}".format("../images/" + image_dir + "/" ,file, 0)
            print string
            out.write(string + '\n')

# print (get_color_from_cclass(0))


create_listfile("pvr_images")
#convert_to_lmdb("CaffeImage_w_labels")