from templates import Tstrings as temps

#Object generator for scene files for povray

#6-7 shapes cone plane shpere cube cylinder , 2D versions
#Different objects rotation position finish
#directional and point lights

class ObjectCreator:
    def __init__(self):
        self.scene = temps.getTemplateString("init")
        # Vector string for fomatting use
        self.VString = "<{0},{1},{2}>"

    def addSphere(self, xyz, size, color, rot=None):
        pos = self.Vstring.format(*xyz)
        rotStr = self.getRotStr(rot)
        str = temps.getTemplateString("sphere").format(position=pos, color=color, size=size, rotString=rotStr)
        self.scene += str

    def addRectPrism(self, xyz, lwh, color, rot=None):
        rotStr = self.getRotStr(rot)
        pos = self.VString.format(*xyz)
        nHalfSizes = self.VString.format(*[-x / 2 for x in lwh])
        sizes = self.VString.format(*lwh)
        str = temps.getTemplateString("rectPrism").format(position=pos, color=color, endPoint=sizes, halfPosition=nHalfSizes, rotString=rotStr)
        self.scene += str

    # Use this to add cylinders by r1=r2
    def addCone(self, xyz, xyz2, r1, r2, color, rot=None):
        rotStr = self.getRotStr(rot)
        center_1 = self.VString.format(*xyz)
        center_2 = self.VString.format(*xyz2)
        str = temps.getTemplateString("cone").format(center_1=center_1,center_2=center_2,radius_1=r1,radius_2=r2,color=color,rotStr=rotStr)
        self.scene += str

    def addPlane(self, xyz, lw, color, rot=None):
        lw.append(0.1)
        self.addRectPrism(xyz,lw,color, rot)

    # Points at origin by default, has some other parameters set by default as well
    def addPointLight(self, xyz, color):
        pos = self.VString.format(*xyz)
        colors = "color red {0} green {1} blue {2}".format(*color)
        str = "light_source {{ {0} {1} }} \n".format(pos, colors)
        self.scene += str

    # Create povray string for rotation with clock param embedded
    def getRotStr(self, rot):
        if rot != None:
            # Rotation amounts by clock cycle
            rotScale = list((str(i) + "*clock")for i in rot)
            rotStr = "<{},{},{}>".format(*rotScale)
            return rotStr
        else:
            return ""
