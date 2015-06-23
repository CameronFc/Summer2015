#python file to generate scene files
from subprocess import call
from subprocess import check_output
from PIL import Image
from scipy import misc
import pprint
import time
import os
import math
import numpy.random as rng
import numpy as np
import matplotlib.pyplot as plt

#Grab a generic scene file
#Write to its contents some of the variables listed for it
#Call the process to render the scene file
#get the resulting image file
#note that with this version we want to dump all of our scene files in the scene folder
#unfortunately, there are no switches that allow disabling of the banner text produced during each render.
#ok, so now we need a way to actually generate the files based on the inputs that are written within this program.
#We need a system that allows us to save the proper type of the created scenes. We to convert the created image files
#into the kind of data set that alllows for manipulatation in the system similar to the MNIST dataset.
#want to shorten rendering time.
#need some way to get rid of images that are case in shadow, i.e not bright


class dirs:
    imageDirectory = "../images/"
    sceneDirectory = "../scenes/"
    templateDirectory = "../templates/"
    index = "../index.txt"
    imageExt = ".png"
    path = os.getcwd() + "/"

def typeSwitcher(x):
    return {
        '0': "training/",
        '1': "tests/",
        '2': "validation/"
    }.get(str(x),"")

classes = {
        'sphere': 0,
        'cube': 1
    }

def intToClass(int):
    for name in classes:
        if classes[name] == int:
            return name

def classToInt(className):
    return classes.get(className, 2)


#print(path)

#note that we actually have to have this directory existing, else pov ray will not be able to save the image
