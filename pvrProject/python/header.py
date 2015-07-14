import os

# Routing to find files quickly based on python variables
class dirs:
    imageDirectory = "../images/"
    imageExt = ".tga"
    sceneDirectory = "../scenes/"
    sceneExt = ".pov"
    templateDirectory = "../templates/"
    templateExt = ".txt"
    savedDataDirectory = "../netParams/"
    savedDataExt = ".pkl"
    index = "../index.txt"
    indices = "../indices/"
    pickle = "data.pkl"
    settings = "../settings.ini"
    path = os.getcwd() + "/"

def typeSwitcher(x):
    return {
        'PTRAIN': "training/",
        'PTESTS': "tests/",
        'PVALID': "validation/"
    }.get(str(x),"")

#Are we creating an animated scene?
def animSwitcher(x):
    return {
        'PSTATIC': "",
        'PANIM': "anim/"
    }.get(str(x),"")


# Old header tools
classes = {
        'sphere': 0,
        'cube': 1
    }

def intToClass(int):
    for name in classes:
        if classes[name] == int:
            return name

def classToInt(className):
    return classes.get(className, 2)

