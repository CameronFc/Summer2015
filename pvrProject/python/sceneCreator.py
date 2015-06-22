from numpy import random as rng
from header import dirs
from header import typeSwitcher

def createScene(sceneName, templateName, type=0):
    #TODO replace line below
    x,y,z = rng.randint(-10,10),rng.randint(-10,10),rng.randint(-10,10)

    with open(dirs.path + dirs.templateDirectory + templateName + ".pov", "r") as template:
        with open("./" + dirs.sceneDirectory + typeSwitcher(type) + sceneName + ".pov", "w") as out:
            for line in template:
                out.write(line)
            out.write("light_source { <" + str(x) +","+ str(y) +","+ str(z) +"> color red 1 green 1 blue 1 }")

def clearIndex():
    with open(dirs.path + dirs.index, "w") as out:
        out.write("")

def writeToIndex(pair):
    with open(dirs.path + dirs.index, "a") as out:
        out.write(pair + "\n")