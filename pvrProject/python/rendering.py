#can include -GA switch to make things faster
import time
import math
import os
from header import dirs
from header import typeSwitcher
from header import animSwitcher
from header import dumpSwitcher
from subprocess import call
from subprocess import check_output
from subprocess import STDOUT
from format import getDesiredFiles

#TODO: Make this actually verobose instead of just errors
#TODO: Fix this renderer init garbage!
verboseRender = True

class Renderer:
    def __init__(self, frames=None, clockFinal=0):

        #Can't include +KFF-1 in command line as this appends '1' to the end of every file
        frameParam = (
            "+KFF" + str(frames) if frames != None
            else ""
        )
        self.callArrayOptions = [
                            "-GA",
                            frameParam
                         ]
        self.animType = (
            "PSTATIC" if frames == None
            else "PANIM"
        )

    def renderImages(self, files, type):
        renderCount = 0
        renderTotal = 0
        # Want the files to be in the dump for animations, static otherwise
        for fileName in files:
            if fileName[-(len(dirs.sceneExt)):] == dirs.sceneExt:
                renderTotal += 1
        startTime = time.time()
        for fileName in files:
                if(fileName[-(len(dirs.sceneExt)):] == dirs.sceneExt):
                    sceneName = fileName[:-4]
                    callArray = ["povray",
                                 dirs.path + dirs.settings,
                                 "./" + dirs.sceneDirectory + typeSwitcher(type) + fileName,
                                 "+O" + dirs.path + dirs.imageDirectory + typeSwitcher(type) + dirs.dump + sceneName
                                 ] + (self.callArrayOptions)
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

    def appendImages(self, name, type="PTRAIN"):
        path = dirs.imageDirectory + typeSwitcher(type)
        # Only appends files with name as prefix
        files, fileNames = getDesiredFiles("./" + path + dirs.dump, name)
        print("Num files appended: {}".format(len(files)))
        # Put the concatenated images into the anim directory
        commandArray = ["convert", "+append", path + animSwitcher(self.animType) + name + dirs.imageExt]
        commandArray[1:1] = files
        returnCode = call(commandArray)
        if returnCode != 0:
            print("Fatal error during image concatenation!")