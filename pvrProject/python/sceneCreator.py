from numpy import random as rng
import numpy as np
from header import dirs
from header import typeSwitcher
import objectCreator as oc

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

def writeToIndex(pair):
    with open(dirs.path + dirs.index, "a") as out:
        out.write(pair + "\n")

def ltlDebugScene(sceneName, type=0):
    with open("./" + dirs.sceneDirectory + typeSwitcher(type) + sceneName + ".pov", "w") as out:
        objC = oc.objectCreator()
        objC.addSphere([0.0,0.0,0.0], 2, [1,0,0])
        objC.addPointLight([3.0,3.0,3.0], [1,1,1])
        out.write(objC.scene)