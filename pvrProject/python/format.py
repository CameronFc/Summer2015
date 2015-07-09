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
    def __init__(self, animType="PSTATIC"):
        self.animType = animType

    def readImage(self, name, type):
        return misc.imread(dirs.imageDirectory + typeSwitcher(type) + animSwitcher(self.animType) + name)

    def displayImage(self, image):
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()

    def getClass(self, imageName):
        #Use some kind of binary search on each line of the index to find the value
        #We might also want to pickle the data if it is convenient
        #return the class of the object if its name is found, otherwise expel error
        for scene in unPickleIndex():
            #print("Scene name of pickled scene", scene.get("name"))
            if scene.get("name") == imageName:
                return scene
        print("Failed to locate class data of image " + imageName)

    def getDesiredFiles(self, path, name):
        files = os.listdir(path)
        desiredFiles = []
        if name != "":
            for file in files:
                if name in file:
                    desiredFiles.append(file)
        else:
            desiredFiles = files
        return desiredFiles

    #TODO: Make sure that the files are all of the same dimension before constructing the image array
    #Have some way to specify what images you want to use to construct the array instead of just every png as it is now
    #Ok, now it uses the specified image extension properly
    def getAllImages(self, type, name="", fileLimit=None):
        if fileLimit == None:
            fileLimit = sys.maxsize
        desiredFiles = self.getDesiredFiles(dirs.path + dirs.imageDirectory + typeSwitcher(type) + animSwitcher(self.animType), name)
        imageCount = 0
        first = True
        baseImage = []
        #Count all of the images and create a base image to gather array sizes
        for fileName in desiredFiles:
            if(fileName[-(len(dirs.imageExt)):] == dirs.imageExt):
                imageCount += 1
                if first:
                    baseImage = self.readImage(fileName, type)
                    first = False

        #print(baseImage.shape)
        dims = list(baseImage.shape)
        dims.insert(0,imageCount)
        #print (dims)
        #imageArray = np.zeros(dims, dtype=np.int)
        imageArray = []
        #print(imageArray)
        classArray = []
        nameList = []
        #Use this to change the maximum number of images that can be feed into the 'net,
        # no matter how many images exist in the folder of the same name type
        processed = 0

        print("Loading image files...")
        for index, fileName in enumerate(desiredFiles):
            #print(fileName[-4:])
            if(fileName[-4:] == dirs.imageExt and processed < fileLimit):
                #print(np.array(readImage(fileName)))
                #print(dir(readImage(fileName)))
                imageArray.append(self.readImage(fileName, type).flatten())
                classArray.append(self.getClass(fileName[:-4]))
                nameList.append(fileName)
                processed += 1
                #print(classArray[index - 1])
                #imageArray = np.append(imageArray,readImage(fileName), axis = 1)
                #displayImage(readImage(fileName))

        print("COMPLETED: Loading image files ({})".format(processed))
        #print(imageArray)
        return imageArray, classArray, nameList

#imageArray, classArray, nameList = getAllImages(0)

#print(type(readImage("cube0.png")))
#displayImage(imageArray[0])
#print(type(imageArray[0]))
#print((imageArray[0].shape))

#print(nameList[0])
#for i in range(len(imageArray)):
#    displayImage(imageArray[i])
#displayImage(readImage("cube0.png"))

#print(readImage("genesis9.png")[60,43])
#print(imageArray[10][60,43])
#formatImageData(im, 0)

