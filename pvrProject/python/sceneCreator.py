from numpy import random as rng
from header import dirs
from header import typeSwitcher
import objectCreator as oc
import numpy as np
import pickle


def createScene(sceneName, templateName, type=0):
    #TODO replace line below
    numLights = 3
    x = np.zeros(3)
    y = np.zeros(3)
    z = np.zeros(3)
    for i in range(numLights):
        x[i],y[i],z[i] = rng.randint(-10,10),rng.randint(-10,10),rng.randint(-10,10)

    with open(dirs.path + dirs.templateDirectory + templateName + ".pov", "r") as template:
        with open("./" + dirs.sceneDirectory + typeSwitcher(type) + sceneName + ".pov", "w") as out:
            for line in template:
                out.write(line)
            for i in range(numLights):
                out.write("light_source { <" + str(x[i]) +","+ str(y[i]) +","+ str(z[i]) +"> color red 1 green 1 blue 1 } \n")

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

def pickleToIndex(dict):
    with open(dirs.path + dirs.pickle, mode="a+b") as out:
        pickle.dump(dict, out)

def ltlDebugScene(sceneName, type=0):
    objC = oc.objectCreator()
    objC.addSphere([0.0,0.0,0.0], 2, [1,0,0])
    xyz = np.zeros(3)
    for index, element in enumerate(xyz):
        xyz[index] = rng.randn(1) * 10
    objC.addPointLight(xyz, [1,1,1])
    dict = {"objects" : {"light": {"position" : xyz}}, "name" : sceneName}
    sceneFromObjCreator(sceneName, objC, dict, type)

##################################################################################################
##Holdovers
def getxyz(scale):
    xyz = np.zeros(3)
    for index, element in enumerate(xyz):
        xyz[index] = rng.randn(1) * scale
    return xyz

def cubeScene(sceneName, type=0):
    objC = oc.objectCreator()
    objC.addRectPrism([0.0,0.0,0.0], [1.5,1.5,1.5], [1,0,0])
    for i in range(3):
        xyz = getxyz(10)
        objC.addPointLight(xyz, [1,1,1])
    dict = {"class" : 1, "name" : sceneName}
    sceneFromObjCreator(sceneName, objC, dict, type)

def sphereScene(sceneName, type=0):
    objC = oc.objectCreator()
    objC.addSphere([0.0,0.0,0.0], 2, [1,0,0])
    for i in range(3):
        xyz = getxyz(10)
        objC.addPointLight(xyz, [1,1,1])
    dict = {"class" : 0, "name" : sceneName}
    sceneFromObjCreator(sceneName, objC, dict, type)
##################################################################################################

def sceneFromObjCreator(sceneName, objC, dict, type):
    with open("./" + dirs.sceneDirectory + typeSwitcher(type) + sceneName + ".pov", "w") as out:
        pickleToIndex(dict)
        out.write(objC.scene)

def unPickleIndex():
    with open(dirs.path + dirs.pickle, "rb") as file:
        lst = []
        while 1:
            try:
                lst.append(pickle.load(file))
            except EOFError:
                break
        #st.append(pickle.load(file))
        return lst

def clearPickleIndex():
    with open(dirs.path + dirs.pickle, "w") as file:
        file.write("")

#clearIndex()
#writeToIndex("Cube","[1,1,1]")

#ltlDebugScene("aDebug")
#print(unPickleIndex())
