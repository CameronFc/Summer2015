from header import dirs
from header import typeSwitcher
import header
import rendering as r
from logistic import Logistic
import sceneCreator as sc
import format
import alteration as alt
import os
import sys

#TODO: Clean up defunct tests

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
        type = "PTESTS"
        if(typeSwitcher(type) == ""):
            print("ERROR: Invalid type: {}".format(type))
            sys.exit(0)
        frames = 20
        numAnims = 20
        # Use this to vary the name of the images we want to create
        index_name = "VaryingSizeCubes"
        sc.clearPickleIndex(type, index_name)
        for i in range(numAnims):
            #need a delimiter to separate scene# from frame#
            delim = "D"
            name = index_name + str(i) + delim
            #bugfix
            sc.third_animation(name, type, index_name)
            # sc.createBasicAnimation(name, 0)
            renderer = r.Renderer(frames=frames)
            renderer.renderByName(name, type)
            renderer.appendImages(name, type)
        print("Completed Anim rendering")

#TODO: Clean up code so things are readable, in correct pythonic format in free time
#TODO: Change to completely uncompressed targa format, disallow unlike images
#TODO: Changing number of frames requires the deletion of all images in the root type directory! (bad)
#TODO: Naming convention is also different depending on the amount of files produced, should change (ex. 1 vs 01 vs 001)

#tests.createScenes()
#tests.createLightScenes()
#tests.createAnimScenes()
tests.CARAS()
#tests.renderLights()
#tests.render()
#tests.convertAllToGreyScale()
#tests.llnetAll()
