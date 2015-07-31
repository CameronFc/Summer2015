global_settings {
  assumed_gamma 2.2
}
camera {
   location  <0, 0, -3.5>
   direction <0, 0, 1.2071>
   look_at   <0, 0, 0>
}
box { <0.0, 0.0, 0.0>, <1.5,1.5,1.5>
    finish {
       ambient 0.6
       diffuse 0.8
       phong 1
    }
    pigment { color red 1 green 0 blue 1 }
    rotate <-20, 30, 0>
}
light_source { <4,4,-4> color red 1 green 1 blue 1 }