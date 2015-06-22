from header import dirs
from header import typeSwitcher
from rendering import renderAll
from rendering import renderFile
import sceneCreator as sc
import format as f
import os
from logistic import logistic
from format import readImage
from format import displayImage

#createScene("genesis", "box")

class tests:
    @staticmethod
    def createScenes():
        sc.clearIndex()
        for i in range(100):
            name = "sphere" + str(i)
            sc.createScene(name, "sphere")
            sc.writeToIndex(name + " sphere")
        for i in range(100):
            name = "cube" + str(i)
            sc.createScene(name, "cube")
            sc.writeToIndex(name + " cube")
        print("Completed")

    @staticmethod
    def render():
        renderAll(0)
        print("Completed")

    @staticmethod
    def logisticAll():
        imageArray, classArray, nameList = f.getAllImages(0)
        classifier = logistic(imageArray, classArray, nameList)
        classifier.beginTraining()
        #for i in range(len(imageArray)):
        #    classifier.classify(imageArray[i])
        #classifier.classify(readImage("TestSphere.png", 1), "TestSphere.png")
        testImageFiles = os.listdir(dirs.path + dirs.imageDirectory + typeSwitcher(1))
        for file in testImageFiles:
            classifier.classify(readImage(file, 1), file)

    @staticmethod
    def CARATS():
        for i in range(100):
            sc.createScene("TestSphere" + str(i), "sphere", 1)
        renderAll(1)

#tests.createScenes()
#tests.render()
tests.logisticAll()
#tests.CARATS()
