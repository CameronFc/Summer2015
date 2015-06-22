// Persistence Of Vision raytracer version 3.1	sample file.
// File by Alexander Enzmann
//
// -w320 -h240
// -w800 -h600 +a0.3

#version 3.6;
global_settings { 
  assumed_gamma 2.2
}

camera {
   location  <0, 0, -8>
   direction <0, 0, 1.2071>
   look_at   <0, 0, 0>
}

box { <-1.0, -1.0, -1.0>, <1.0, 1.0, 1.0>
    finish {
       ambient 0.2
       diffuse 0.8
       phong 1
    }
    pigment { color red 1 green 0 blue 0 }
 
    rotate <-20, 30, 0>
}

