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
        sc.clearIndex()
        for i in range(100):
            name = "sphere" + str(i)
            #sc.createScene(name, "sphere")
            sc.ltlDebugScene(name)
            sc.writeToIndex(name + " sphere")
        for i in range(100):
            name = "cube" + str(i)
            #sc.createScene(name, "cube")
            sc.ltlDebugScene(name)
            sc.writeToIndex(name + " cube")
        print("Completed creating scenes")

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
            sc.createScene(name, "sphere", 1)
            sc.writeToIndex(name + " sphere")
        renderAll(1)

    @staticmethod
    def convertAllToGreyScale():
        alt.allImagesToGreyScale(0)
        alt.allImagesToGreyScale(1)


#TODO: Improve how indexing works so were don't have to run createscenes->CARATS in that order to maintain the index
#TODO: Make grey-scale test procedure
#TODO: fix .DS_Store interaction with file counting
#TODO: make classification classes more integrated into project instead of 1's and 0's


tests.createScenes()
#tests.render()
#tests.CARATS()
#tests.convertAllToGreyScale()
#tests.logisticAll()

