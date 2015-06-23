#can include -GA switch to make things faster
import time
import math
import os
from header import dirs
from header import typeSwitcher
from subprocess import call
from subprocess import check_output
from subprocess import STDOUT

#TODO: Mae this actually verobose instead of just errors
verboseRender = True



def renderImages(files, type):
    renderCount = 0
    renderTotal = len(files) #.DS_Store must be excluded? -1 if it is there
    startTime = time.time()
    for fileName in files:
            if(fileName[-4:] == ".pov"):
                sceneName = fileName[:-4]
                returncode = call([dirs.path + "../Povray/povray", "./" + dirs.sceneDirectory + typeSwitcher(type) + fileName,
                                   "+O" + dirs.path + dirs.imageDirectory + typeSwitcher(type) + sceneName,
                                   "-GA"], stdout=open(os.devnull, "w"),  stderr=STDOUT)
                if returncode != 0 and verboseRender:
                    print("Fatal Error during rendering: " + str(returncode)
                         ," On scene with name: " + sceneName)
                    break
                else:
                    renderCount += 1
                    if(renderCount % math.ceil(math.sqrt(renderTotal)) == 0
                           or renderCount == renderTotal or renderCount == 1):
                        print("Rendered " + str(renderCount) + "/" + str(renderTotal))
            #endif
    #EndFor
    print("Rendering time: {0} seconds".format(str(time.time() - startTime)))


def renderAll(type):
    files = os.listdir(dirs.path + dirs.sceneDirectory + typeSwitcher(type))
    renderImages(files, type)

def renderFile(fileName, type):
    renderImages([fileName], type)