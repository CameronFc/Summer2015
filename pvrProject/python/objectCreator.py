from templates import Tstrings as temps

#Object generator for scene files for povray

#Must need some kind of transfer mechanism to pass to a scene creator to save a total scene file
# return type: some kind of string is appropriate...
#

#6-7 shapes cone plane shpere cube cylinder , 2D versions
#Different objects rotation position finish
#directional and point lights

class ObjectCreator:
    def __init__(self):
        self.scene = temps.getTemplateString("init")
        #Vector string for fomatting use
        self.VString = "<{0},{1},{2}>"

    def addSphere(self, xyz, size, color, rot=None):
        pos = self.Vstring.format(*xyz)
        rotStr = self.getRotStr(rot)
        str = temps.getTemplateString("sphere").format(position=pos, color=color, size=size, rotString=rotStr)
        self.scene += str
        #print(self.scene)

    def addRectPrism(self, xyz, lwh, color, rot=None):
        rotStr = self.getRotStr(rot)
        pos = self.VString.format(*xyz)
        nHalfSizes = self.VString.format(*[-x / 2 for x in lwh])
        sizes = self.VString.format(*lwh)
        str = temps.getTemplateString("rectPrism").format(position=pos, color=color, endPoint=sizes, halfPosition=nHalfSizes, rotString=rotStr)
        self.scene += str
        #print(self.scene)

    def addPlane(self, xyz, lw, color, rot=None):
        lw.append(0.1)
        self.addRectPrism(xyz,lw,color, rot)

    #points at origin by default, has some other parameters set by default as well
    def addPointLight(self, xyz, color):
        pos = self.VString.format(*xyz)
        colors = "color red {0} green {1} blue {2}".format(*color)
        str = "light_source {{ {0} {1} }} \n".format(pos, colors)
        self.scene += str
        #print(self.scene)

    def getRotStr(self, rot):
        if rot != None:
            #Rotation amounts by clock cycle
            rotScale = list((str(i) + "*clock")for i in rot)
            rotStr = "<{},{},{}>".format(*rotScale)
            return rotStr
        else:
            return ""
        #print(")


#objC = objectCreator()
#objC.addSphere([0,1,2], 3, [1.0,0.5,0.3])
#objC.addRectPrism([0,1,2], [1,1,1], [1.0,0.5,0.3])
#objC.addPlane([0,1,2], [3,4], [1.0,0.5,0.3])
#objC.addPointLight([3,3,3], [0.1,0.2,0.5])
