

#Object generator for scene files for povray

#Must need some kind of transfer mechanism to pass to a scene creator to save a total scene file
# return type: some kind of string is appropriate...
#

#6-7 shapes cone plane shpere cube cylinder , 2D versions
#Different objects rotation position finish
#directional and point lights
class objectCreator:
    def __init__(self):
        self.scene = ""
        self.scene+= """global_settings {
  assumed_gamma 2.2
}
camera {
   location  <0, 0, -8>
   direction <0, 0, 1.2071>
   look_at   <0, 0, 0>
}\n"""


    def addSphere(self, xyz, size, color):
        pos = "<{0},{1},{2}>".format(*xyz)
        str = """sphere {{ {0}, {2}
    finish {{
      ambient 0.2
      diffuse 0.8
      phong 1
    }}
    pigment {{ color red {1[0]} green {1[1]} blue {1[2]} }}
}}\n""".format(pos, color, size)
        self.scene += str
        #print(self.scene)

    def addRectPrism(self, xyz, lwh, color):
        pos = "<{0},{1},{2}>".format(*xyz)
        nHalfSizes = "<{0},{1},{2}>".format(*[-x / 2 for x in lwh])
        sizes = "<{0},{1},{2}>".format(*lwh)
        str = """box {{ <0.0, 0.0, 0.0>, {2}
    finish {{
       ambient 0.2
       diffuse 0.8
       phong 1
    }}
    pigment {{ color red {1[0]} green {1[1]} blue {1[2]} }}
    translate {0}
    translate {3}
    rotate <-20, 30, 0>
}}\n""".format(pos, color, sizes, nHalfSizes)
        self.scene += str
        #print(self.scene)

    def addPlane(self, xyz, lw, color):
        lw.append(0.1)
        self.addRectPrism(xyz,lw,color)

    #points at origin by default, has some other parameters set by default as well
    def addPointLight(self, xyz, color):
        pos = "<{0},{1},{2}>".format(*xyz)
        colors = "color red {0} green {1} blue {2}".format(*color)
        str = "light_source {{ {0} {1} }} \n".format(pos, colors)
        self.scene += str
        #print(self.scene)

objC = objectCreator()
#objC.addSphere([0,1,2], 3, [1.0,0.5,0.3])
objC.addRectPrism([0,1,2], [1,1,1], [1.0,0.5,0.3])
objC.addPlane([0,1,2], [3,4], [1.0,0.5,0.3])
objC.addPointLight([3,3,3], [0.1,0.2,0.5])

