from header import dirs
from header import typeSwitcher
from scipy import misc
from scipy import ndimage
import os



rgbSum = [0,0,0]
def convertImage(image):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            rgbSum[0] += image[x,y][0]
            rgbSum[1] += image[x,y][1]
            rgbSum[2] += image[x,y][2]
    print(rgbSum)

#convertImage(misc.imread(dirs.path + dirs.imageDirectory + typeSwitcher(type) + "genesis6.png"))

def fileImageToGreyScale(filename, type):
    imURI = dirs.path + dirs.imageDirectory + typeSwitcher(type) + filename
    image = ndimage.imread(imURI, flatten=True)
    misc.imsave(imURI, image)

#fileImageToGreyScale("Sphere0.png", 0)

def allImagesToGreyScale(type):
    files = os.listdir(dirs.path + dirs.imageDirectory + typeSwitcher(type))
    for file in files:
        if file[-4:] == dirs.imageExt:
            fileImageToGreyScale(file, type)
    print("COMPLETED: Converted all " + typeSwitcher(type)[:-1] + " images to greyscale.")

