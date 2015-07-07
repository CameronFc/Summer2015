#can include -GA switch to make things faster
import time
import math
import os
from header import dirs
from header import typeSwitcher
from header import animSwitcher
from subprocess import call
from subprocess import check_output
from subprocess import STDOUT
import format

#TODO: Mae this actually verobose instead of just errors
verboseRender = True

class Renderer:
    def __init__(self, frames=-1, clockFinal=0):
        self.callArrayOptions = [
                            "-GA",
                            "+KFF" + str(frames)
                         ]
        self.animType = (
            0 if frames == -1
            else 1
        )

    def renderImages(self, files, type):
        renderCount = 0
        renderTotal = 0
        for fileName in files:
            if fileName[-4:] == ".pov":
                renderTotal += 1
        startTime = time.time()
        for fileName in files:
                if(fileName[-4:] == ".pov"):
                    sceneName = fileName[:-4]
                    callArray = ["povray", "./" + dirs.sceneDirectory + typeSwitcher(type) + fileName,
                                 "+O" + dirs.path + dirs.imageDirectory + typeSwitcher(type) + sceneName] + (self.callArrayOptions)
                    returncode = call(callArray, stdout=open(os.devnull, "w"),  stderr=STDOUT)
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

    def renderAll(self, type):
        files = os.listdir(dirs.path + dirs.sceneDirectory + typeSwitcher(type))
        self.renderImages(files, type)

    def renderByName(self, name, type):
        files = os.listdir(dirs.path + dirs.sceneDirectory + typeSwitcher(type))
        desiredFiles = []
        for file in files:
            if name in file:
                desiredFiles.append(file)
        self.renderImages(desiredFiles, type)


    def renderFile(self, fileName, type):
        self.renderImages([fileName], type)

    def appendImages(self, name, type=0):
        formatter = format.Formatter()
        path = dirs.imageDirectory + typeSwitcher(type)
        files = formatter.getDesiredFiles("./" + path, name)
        commandFiles = list((path + file)for file in files)
        #print(commandFiles)
        print("Num files appended: {}".format(len(commandFiles)))
        #put the concatenated images into the anim directory
        commandArray = ["convert", "+append", path + animSwitcher(self.animType) + name + dirs.imageExt]
        commandArray[1:1] = commandFiles
        #print(commandArray)
        returnCode = call(commandArray)
        if returnCode != 0:
            print("Fatal error during image concatenation!")