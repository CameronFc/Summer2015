from header import dirs
import theano
import theano.tensor as T
from numpy import random as rng
import numpy as np

def save_test_results():
    pass

def display():
    pass

# TODO: Remove test material. This is not a sandbox for theano.

# w = theano.shared(rng.randn(3,3,3), name="w")
#
# x = T.tensor3(name='x')
# out = (w * x).sum(axis=0)
#
# train = theano.function(
#     inputs=[x],
#     outputs=out
# )
#
# input = rng.randn(3,3,3)
#
# print(train(input))
dims = [3,4,4]
padding_width = 1

reshaped_image = rng.randn(3,4,4)
depth = dims[0]
len_x = dims[1]
len_y = dims[2]

padded_image = np.zeros((depth, len_x + 2 * padding_width, len_y + 2 * padding_width))

for i in range(depth):
    layer = reshaped_image[i,:,:]
    padded_image[i,padding_width:len_x + padding_width,padding_width:len_y + padding_width] = layer

print(padded_image)

