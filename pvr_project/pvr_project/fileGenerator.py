from header import typeSwitcher
import rendering as r
import sceneCreator as sc
import alteration as alt
import sys


class FileGenerator:

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
        numAnims = 100
        # Use this to vary the name of the images we want to create
        index_name = "StaticCube"
        sc.clearPickleIndex(type, index_name)
        for i in range(numAnims):
            #need a delimiter to separate scene# from frame#
            delim = "D"
            name = index_name + str(i) + delim
            #bugfix
            sc.secondAnimation(name, type, index_name)
            # sc.createBasicAnimation(name, 0)
            renderer = r.Renderer(frames=None)
            renderer.renderByName(name, type)
            renderer.appendImages(name, type)
        print("Completed Anim rendering")

#TODO: Clean up code so things are readable, in correct pythonic format in free time
#TODO: Change to completely uncompressed targa format, disallow unlike images
#TODO: Changing number of frames requires the deletion of all images in the root type directory! (bad)
#TODO: Naming convention is also different depending on the amount of files produced, should change (ex. 1 vs 01 vs 001)
#TODO: Use .append and generators instead of += for lists

FileGenerator.CARAS()
#tests.convertAllToGreyScale()