from header import dirs
from header import typeSwitcher
from scipy import misc
import matplotlib.pyplot as plt
import os
import numpy as np

#Convert the data of a given image into MNIST format
def formatImageListData(images, trueClassList):
    return np.array([images,trueClassList])

def readImage(name, type):
    return misc.imread(dirs.imageDirectory + typeSwitcher(type) + name)

def displayImage(image):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()

def getClass(imageName):
    #Use some kind of binary search on each line of the index to find the value
    #We might also want to pickle the data if it is convenient
    #return the class of the object if its name is found, otherwise expel error
    with open(dirs.path + dirs.index, "r") as out:
        for line in out:
            if line[:len(imageName)] == imageName:
                return line[(len(imageName)+1):-1]

def classToInt(className):
    if className == 'sphere':
        return 0
    if className == 'cube':
        return 1
    return 2

#TODO: Make sure that the files are all of the same dimension before constructing the image array
#Have some way to specify what images you want to use to construct the array instead of just every png as it is now
#Ok, now it uses the specified image extension properly
def getAllImages(type):
    files = os.listdir(dirs.path + dirs.imageDirectory + typeSwitcher(type))

    imageCount = 0
    first = True
    baseImage = []
    #Count all of the images and create a base image to gather array sizes
    for fileName in files:
        if(fileName[-4:] == dirs.imageExt):
            imageCount += 1
            if first:
                baseImage = readImage(fileName, type)
                first = False

    #print(baseImage.shape)
    dims = list(baseImage.shape)
    dims.insert(0,imageCount)
    #print (dims)
    #imageArray = np.zeros(dims, dtype=np.int)
    imageArray = []
    #print(imageArray)
    classArray = np.zeros(imageCount, dtype=np.int)
    nameList = []

    for index, fileName in enumerate(files):
        #print(fileName[-4:])
        if(fileName[-4:] == dirs.imageExt):
            #print(np.array(readImage(fileName)))
            #print(dir(readImage(fileName)))
            imageArray.append(readImage(fileName, type).flatten())
            classArray[index - 1] = classToInt(getClass(fileName[:-4]))
            nameList.append(fileName)
            #print(classArray[index - 1])
            #imageArray = np.append(imageArray,readImage(fileName), axis = 1)
            #displayImage(readImage(fileName))

    #print(imageArray)
    return imageArray, classArray, nameList

#imageArray, classArray, nameList = getAllImages()

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