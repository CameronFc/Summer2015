from header import dirs
from header import typeSwitcher
import header
import rendering as r
from logistic import Logistic
from llnet import LLnet
import sceneCreator as sc
import format
import alteration as alt
import os

class tests:
    @staticmethod
    def createScenes():
        sc.clearPickleIndex()
        for i in range(100):
            name = "sphere" + str(i)
            sc.sphereScene(name)
        for i in range(100):
            name = "cube" + str(i)
            sc.cubeScene(name)
        print("Completed creating basic scenes")
        #for i in sc.unPickleIndex(): print(i)

    @staticmethod
    def createLightScenes():
        sc.clearPickleIndex()
        for i in range(100):
            name = "light" + str(i)
            sc.ltlDebugScene(name)
        print("Completed creating light scenes")

    @staticmethod
    def render():
        renderer = r.Renderer()
        renderer.renderAll(0)
        print("Completed rendering all " + typeSwitcher(0)[:-1] + " images")

    @staticmethod
    def renderLights():
        renderer = r.Renderer()
        renderer.renderByName("light",0)

    @staticmethod
    def logisticAll():
        formatter = format.Formatter(animType=0)
        classifier = Logistic(*formatter.getAllImages(0))
        classifier.beginTraining()
        #for i in range(len(imageArray)):
        #    classifier.classify(imageArray[i])
        #classifier.classify(readImage("TestSphere.png", 1), "TestSphere.png")
        testImageFiles = os.listdir(dirs.path + dirs.imageDirectory + typeSwitcher(1))
        #get the test images
        imageArray, classArray, nameList = formatter.getAllImages(1)
        classifier.classifyImages(imageArray, classArray, nameList)
        #for file in testImageFiles:
        #    classifier.classify(f.readImage(file, 1), file)

    @staticmethod
    def llnetAll():
        fileLimit = 10
        formatter = format.Formatter(animType="PANIM")
        estimator = LLnet(*formatter.getAllImages("PTRAIN", "TestAnimation", fileLimit))
        estimator.beginTraining()
        estimator.saveParams()
        #imageArray, classArray, names = f.getAllImages(0,"light0")
        #estimator.classify(imageArray)
        #print(classArray[0])

    @staticmethod
    #Create and render all test spheres
    def CARATS():
        for i in range(100):
            name = "TestSphere" + str(i)
            sc.sphereScene(name, 1)
        renderer = r.Renderer()
        renderer.renderAll(1)

    @staticmethod
    def convertAllToGreyScale():
        alt.allImagesToGreyScale(0)
        alt.allImagesToGreyScale(1)

    @staticmethod
    def CARAS():
        sc.clearPickleIndex()
        frames = 20
        numAnims = 100
        for i in range(numAnims):
            #need a delimiter to separate scene# from frame#
            delim = "D"
            name = "TestAnimation" + str(i) + delim
            #bugfix
            sc.createBasicAnimation(name, "PTRAIN")
            # sc.createBasicAnimation(name, 0)
            renderer = r.Renderer(frames=frames)
            renderer.renderByName(name, "PTRAIN")
            renderer.appendImages(name, "PTRAIN")
        print("Completed Anim rendering")

#TODO: Make sure changes, now with templating, works (objC)

#tests.createScenes()
#tests.createLightScenes()
#tests.createAnimScenes()
#tests.CARAS()
#tests.renderLights()
#tests.render()
#tests.CARATS()
#tests.convertAllToGreyScale()
#tests.logisticAll()
tests.llnetAll()
