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

# dims = [3,4,4]
# padding_width = 1
#
# reshaped_image = rng.randn(3,4,4)
# depth = dims[0]
# len_x = dims[1]
# len_y = dims[2]
#
# padded_image = np.zeros((depth, len_x + 2 * padding_width, len_y + 2 * padding_width))
#
# for i in range(depth):
#     layer = reshaped_image[i,:,:]
#     padded_image[i,padding_width:len_x + padding_width,padding_width:len_y + padding_width] = layer
#
# print(padded_image)

reshaped_image = T.tensor3(name="reshaped_image")
image_x = T.shape(reshaped_image)[1]
image_y = T.shape(reshaped_image)[2]

stride = 1
rfs_x = 4
rfs_y = 4
depth = 3

filter_dims = [3,4,4]

x_neurons = (image_x - rfs_x)/stride + 1
y_neurons = T.cast((image_y - rfs_y)/stride + 1, 'int64')

#out_array = theano.shared(np.array((x_neurons * y_neurons, rfs_x * rfs_y * depth)))
out_array = T.matrix()

# def extend_array(array, ri , value):
#     col = ri[:,x_begin:x_end,y_begin:y_end].flatten()
#     T.set_subtensor(array[step_x * step_y,:], col)

# for step_x in range(x_neurons):
#     for step_y in range(y_neurons):
#         x_begin = step_x * stride
#         x_end = step_x * stride + rfs_x
#         y_begin = step_y * stride
#         y_end = step_y * stride + rfs_y
#         col = reshaped_image[:,x_begin:x_end,y_begin:y_end].flatten()
#         T.set_subtensor(out_array[step_x * step_y,:], col)

# results, updates = theano.scan(fn=extend_array,
#                                outputs_info=None,
#                                sequences=[],
#                                non_sequences=[])


# func = theano.function(
#     inputs=[reshaped_image],
#     outputs=out_array
# )


#print(func(rng.randn(3,100,100)))

#rng.randn(3,100,100)


