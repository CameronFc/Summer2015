import os

class Dirs:
    core_path = "../net_sources/"
    data = "../data/"


def get_paths():
    try:
        paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except:
        paths = ["Failed to get paths"]

    print "Make sure caffe is in one of the following:"

    for i in paths:
        print i
