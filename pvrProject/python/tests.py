from header import dirs
from header import typeSwitcher
import header
from rendering import renderAll
from rendering import renderFile
from logistic import logistic
import sceneCreator as sc
import format as f
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
        print("Completed creating scenes")
        #for i in sc.unPickleIndex(): print(i)

    @staticmethod
    def render():
        renderAll(0)
        print("Completed rendering all " + typeSwitcher(0)[:-1] + " images")

    @staticmethod
    def logisticAll():
        imageArray, classArray, nameList = f.getAllImages(0)
        classifier = logistic(imageArray, classArray, nameList)
        classifier.beginTraining()
        #for i in range(len(imageArray)):
        #    classifier.classify(imageArray[i])
        #classifier.classify(readImage("TestSphere.png", 1), "TestSphere.png")
        testImageFiles = os.listdir(dirs.path + dirs.imageDirectory + typeSwitcher(1))
        #get the test images
        imageArray, classArray, nameList = f.getAllImages(1)
        classifier.classifyImages(imageArray, classArray, nameList)
        #for file in testImageFiles:
        #    classifier.classify(f.readImage(file, 1), file)

    @staticmethod
    #Create and render all test spheres
    def CARATS():
        for i in range(100):
            name = "TestSphere" + str(i)
            sc.sphereScene(name, 1)
        renderAll(1)

    @staticmethod
    def convertAllToGreyScale():
        alt.allImagesToGreyScale(0)
        alt.allImagesToGreyScale(1)



tests.createScenes()
tests.render()
tests.CARATS()
tests.convertAllToGreyScale()
tests.logisticAll()

