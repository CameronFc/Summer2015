from numpy import random as rng
from header import dirs
from header import typeSwitcher
import objectCreator as oc
import numpy as np
import pickle

def ltlDebugScene(sceneName, type, index_name):
    objC = oc.ObjectCreator()
    objC.addSphere([0.0,0.0,0.0], 2, [1,0,0])
    xyz = np.zeros(3)
    for index, element in enumerate(xyz):
        xyz[index] = rng.randn(1) * 10
    objC.addPointLight(xyz, [1,1,1])
    dict = {"objects" : {"light": {"position" : xyz}}, "name" : sceneName}
    sceneFromObjCreator(sceneName, objC, dict, type, index_name)

def createBasicAnimation(sceneName, type, index_name):
    objC = oc.ObjectCreator()
    rot = list(rng.randint(0,360, 3))
    objC.addRectPrism([0.0,0.0,0.0], [1.5,1.5,1.5], [1,0,0], rot)
    xyz=getxyz(10)
    objC.addPointLight(xyz, [1,1,1])
    dict = {"objects" : {"light": {"position" : xyz}}, "name" : sceneName}
    sceneFromObjCreator(sceneName, objC, dict, type, index_name)

def secondAnimation(sceneName, type, index_name):
    objC = oc.ObjectCreator()
    rot = list(rng.randint(0,360, 3))
    color = getRandColor()
    objC.addRectPrism([0.0,0.0,0.0], [1.5,1.5,1.5], color, rot)
    xyz=getxyz(10)
    objC.addPointLight(xyz, [1,1,1])
    dict = {"objects" :
                [
                    {"light":
                         {"position" : xyz}
                    },
                    {"rect":
                         {"color": color,
                          "color_class": get_color_class(color)}
                    }
                ],
           "name" : sceneName}
    sceneFromObjCreator(sceneName, objC, dict, type, index_name)

def third_animation(scene_name, type, index_name):
    objC = oc.ObjectCreator()
    rot = list(rng.randint(0,360, 3))
    color = getRandColor()
    size = rng.random_sample(1)[0] * 1.5 + 1
    objC.addRectPrism([0.0,0.0,0.0], [size,size,size], color, rot)
    xyz=getxyz(10)
    objC.addPointLight(xyz, [1,1,1])
    dict = {"objects" :
                [
                    {"light":
                         {"position" : xyz}
                    },
                    {"rect":
                         {"color": color,
                          "color_class": get_color_class(color),
                          "size": size}
                    }
                ],
           "name" : scene_name}
    sceneFromObjCreator(scene_name, objC, dict, type, index_name)

##################################################################################################
## Helpers

def getxyz(scale):
    xyz = np.zeros(3)
    for index, element in enumerate(xyz):
        xyz[index] = rng.randn(1) * scale
    return xyz

def get_color_class(color):
    return int(4 * color[0] + 2 *color[1] + color[2])

def sceneFromObjCreator(sceneName, objC, dict, type, index_name):
    with open("./" + dirs.sceneDirectory + typeSwitcher(type) + sceneName + dirs.sceneExt, "w") as out:
        #Save the data
        pickleToIndex(dict, type, index_name)
        #What actually creates the scene file
        out.write(objC.scene)

##################################################################################################
## Holdovers

def getRandColor():
    color = np.zeros(3)
    for index, element in enumerate(color):
        color[index] = rng.randint(0,2)
    # Make a random color if we set as black or white
    if(np.all(color==color[0])):
        index = rng.randint(0,3)
        not_index = (index + 1) % 3
        color[index] = 1
        color[not_index] = 0
    return color

def cubeScene(sceneName, type, index_name):
    objC = oc.ObjectCreator()
    objC.addRectPrism([0.0,0.0,0.0], [1.5,1.5,1.5], [1,0,0])
    for i in range(3):
        xyz = getxyz(10)
        objC.addPointLight(xyz, [1,1,1])
    dict = {"class" : 1, "name" : sceneName}
    sceneFromObjCreator(sceneName, objC, dict, type, index_name)

def sphereScene(sceneName, type, index_name):
    objC = oc.ObjectCreator()
    objC.addSphere([0.0,0.0,0.0], 2, [1,0,0])
    for i in range(3):
        xyz = getxyz(10)
        objC.addPointLight(xyz, [1,1,1])
    dict = {"class" : 0, "name" : sceneName}
    sceneFromObjCreator(sceneName, objC, dict, type, index_name)

##################################################################################################
## INDEXING

def pickleToIndex(dict, type, index_name):
    with open(dirs.path + dirs.indices + typeSwitcher(type) + index_name + dirs.index_ext, mode="a+b") as out:
        pickle.dump(dict, out)

def unPickleIndex(type, index_name):
    with open(dirs.path + dirs.indices + typeSwitcher(type) + index_name + dirs.index_ext, "rb") as file:
        file.seek(0)
        lst = []
        while 1:
            try:
                lst.append(pickle.load(file))
            except EOFError:
                break
        #st.append(pickle.load(file))
        return lst

def clearPickleIndex(type, index_name):
    open(dirs.path + dirs.indices + typeSwitcher(type) + index_name + dirs.index_ext, "w").close()
    print("Cleared {} Index".format(typeSwitcher(type) + index_name + dirs.index_ext))

##################################################################################################
## Old text based:
def clearIndex():
    with open(dirs.path + dirs.index, "w") as out:
        out.write("")

def writeToIndex(*args, d=','):
    with open(dirs.path + dirs.index, "a") as out:
        str = ""
        for x in args:
            str += x + d
        #remove last delimeter
        str = str[:-1]
        out.write(str + "\n")
