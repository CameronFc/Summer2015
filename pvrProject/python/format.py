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

#TODO: Make getDesiredFiles static, it's irrespective of everything

class Formatter:
    def __init__(self, anim_type="PSTATIC"):
        self.anim_type = anim_type

    def readImage(self, pathname):
        return misc.imread(pathname)

    def displayImage(self, image):
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()

    def getClass(self, imageName, type):
        #Use some kind of binary search on each line of the index to find the value
        #We might also want to pickle the data if it is convenient
        #return the class of the object if its name is found, otherwise expel error
        for scene in unPickleIndex(type):
            #print("Scene name of pickled scene", scene.get("name"))
            if scene.get("name") == imageName:
                return scene
        print("Failed to locate class data of image " + imageName)

    def getDesiredFiles(self, path, name):
        files = os.listdir(path)
        desiredFiles = []
        fileNames = []
        if name != "":
            for file in files:
                if name in file:
                    desiredFiles.append(path + file)
                    fileNames.append(file[:-(len(dirs.imageExt))])
        else:
            #TODO: Clean me up
            print("SPECIFIY A NAME!")
            sys.exit(0)
        #    desiredFiles = [(path + file) for file in files]
        return desiredFiles, fileNames

    def getImage(self, name, type):
        return [self.readImage(name, type)], [self.getClass(name)], [name]


    #TODO: Make sure that the files are all of the same dimension before constructing the image array
    #Have some way to specify what images you want to use to construct the array instead of just every png as it is now
    #Ok, now it uses the specified image extension properly
    def get_dataset(self, type="", name="", file_limit=None):
        print("Loading image files...")
        if file_limit == None:
            file_limit = sys.maxsize
        desired_files, names = self.getDesiredFiles(dirs.path + dirs.imageDirectory + typeSwitcher(type) + animSwitcher(self.anim_type), name)
        # Limit the number of classes and images to the filelimit
        file_list, name_list = desired_files[:file_limit], names[:file_limit]
        image_array = []
        class_array = []
        for path_file, image_name in zip(file_list, name_list):
            image_array.append(self.readImage(path_file).flatten())
            class_array.append(self.getClass(image_name, type))
        print("COMPLETED: Loading image files")
        return image_array, class_array
