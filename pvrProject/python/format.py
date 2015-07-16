from header import dirs
from header import typeSwitcher
from header import animSwitcher
from header import classToInt
from scipy import misc
from sceneCreator import unPickleIndex
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

class Formatter:
    def __init__(self, anim_type="PSTATIC"):
        self.anim_type = anim_type

    def getClass(self, imageName, type, index_name):
        for scene in unPickleIndex(type, index_name):
            if scene.get("name") == imageName:
                return scene
        print("Failed to locate class data of image " + imageName)

    #TODO: Make sure that the files are all of the same dimension before constructing the image array
    def get_dataset(self, type="", name="", file_limit=None):
        print("Loading image files...")
        if file_limit == None:
            file_limit = sys.maxsize
        desired_files, names = getDesiredFiles(dirs.path + dirs.imageDirectory + typeSwitcher(type) + animSwitcher(self.anim_type), name)
        if(len(names) == 0):
            print("Could not find any files with type {}, anim_type {}, name {}".format(type,self.anim_type,name))
        # Limit the number of classes and images to the filelimit
        file_list, name_list = desired_files[:file_limit], names[:file_limit]
        image_array = []
        class_array = []
        for path_file, image_name in zip(file_list, name_list):
            image_array.append(readImage(path_file).flatten())
            class_array.append(self.getClass(image_name, type, name))
        print("COMPLETED: Loading image files")
        return image_array, class_array

# Returns files with path pre-appended, and also the raw file names themselves
def getDesiredFiles(path, name):
    files = os.listdir(path)
    desiredFiles = []
    fileNames = []
    if name != "":
        for file in files:
            if name in file:
                desiredFiles.append(path + file)
                fileNames.append(file[:-(len(dirs.imageExt))])
    else:
        print("SPECIFIY A NAME!")
        sys.exit(0)
    return desiredFiles, fileNames

def displayImage(image):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()

def readImage(pathname):
    return misc.imread(pathname)