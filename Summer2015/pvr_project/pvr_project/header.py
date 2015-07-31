import os

# Routing to find files quickly based on python variables
class dirs:
    dump = "dump/"
    imageDirectory = "../images/"
    imageExt = ".jpg"
    index = "../index.txt"
    indices = "../indices/"
    index_ext = ".pkl"
    path = os.getcwd() + "/"
    pickle = "data.pkl"
    savedDataDirectory = "../netParams/"
    savedDataExt = ".pkl"
    sceneDirectory = "../scenes/"
    sceneExt = ".pov"
    settings = "../settings.ini"
    templateDirectory = "../templates/"
    templateExt = ".txt"


def typeSwitcher(x):
    return {
        'PTRAIN': "training/",
        'PTESTS': "tests/",
        'PVALID': "validation/"
    }.get(str(x),"")

#Are we creating an animated scene?
def animSwitcher(x):
    return {
        'PSTATIC': "static/",
        'PANIM': "anim/"
    }.get(str(x),"")

def dumpSwitcher(x):
    return (
        dirs.dump if x == "PANIM"
        else animSwitcher(x)
    )

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

