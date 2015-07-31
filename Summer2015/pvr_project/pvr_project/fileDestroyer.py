from header import dirs
from header import typeSwitcher
import os


def deleteAllScenes(type):
    os.remove(dirs.path + dirs.sceneDirectory + typeSwitcher(type))

def deleteAllImages(type):
    os.remove(dirs.path + dirs.imageDirectory + typeSwitcher(type))

def deleteAll():
    for i in range(3):
        deleteAllScenes(i)
        deleteAllImages(i)

#Sudo this file from command line
#TODO: make this actually work by correctly setting permissions
deleteAll()
