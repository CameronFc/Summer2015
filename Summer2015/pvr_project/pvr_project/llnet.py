import numpy as np
import theano
import theano.tensor as T
import sys
import pickle
import convLayer
from time import strftime
from header import dirs
rng = np.random

# Logistic-linear net to estimate the position of lights
# Note that this file should be completely unmodified during tests, and should
# house no information related to a particular test.

#TODO: Add batching function

#theano.config.compute_test_value = 'warn'

class HiddenLayer:
    def __init__(self, input, dims, num_layers, activation=None):
        # TODO: CHANGE ME BACK
        # wBase = np.ones(*dims)
        # wBase = rng.randn(*dims)
        self.w = theano.shared(1., name='w' + str(num_layers))
        self.b = theano.shared(0. , name='b' + str(num_layers))
        linOutput = T.dot(self.w, input) + self.b
        self.output = (
            linOutput if activation is None
            else activation(linOutput)
        )
        self.params = [self.w,self.b]

# class ConvLayer(HiddenLayer):
#     def __init__(self, input, dims, num_layers, activation=None):
#         HiddenLayer.__init__(input, dims, num_layers, activation=None)
#
#
#         self.rfs = 3 # Receptive field always square
#         self.stride = 1
#         self.zero_padding = 1
#         # We want to pad zeros along the sides
#         # Each neuron in the conv layer is responsible for one receptive field
#         y = 60
#         x = 80
#         self.output = []
#         image = input.reshape((y,x,3))
#         for step_x in range((x - self.rfs)/self.stride + 1):
#             for step_y in range((y - self.rfs)/self.stride + 1):
#                 w_i = theano.shared(rng.randn(self.rfs,self.rfs), name="w_i")
#                 b_i = theano.shared(0., name="b_i")
#                 # Here we want neuron i,j to scan pixels x,x+rfs in width
#                 # and y,y+rfs in height, and also hit the three color parts of the pixels.
#                 # Assume that we have a real 3D image passed into this layer.
#                 x_1 = self.stride * step_x
#                 x_2 = x_1 + self.rfs
#                 y_1 = self.stride * step_y
#                 y_2 = y_1 + self.rfs
#                 activation_layer = T.dot(image[x_1:x_2][y_1:y_2], w_i) + b_i
#                 # Now we need some way to combine the output
#                 self.output.append(activation_layer)
#
#     def im_to_col(self, input, dims):
#         reshaped_image = input.reshape(dims)
#
#
#     # Note this requires format of (depth, x, y)
#     def get_padded_image(self, input, dims, padding_width=1):
#         depth = dims[0]
#         len_x = dims[1]
#         len_y = dims[2]
#         padded_image = np.zeros((depth, len_x + 2 * padding_width, len_y + 2 * padding_width))
#         for i in range(depth):
#             layer = input[i,:,:]
#             padded_image[i, padding_width:padding_width + len_x, padding_width:padding_width + len_y] = layer
#         return(padded_image)

class LLNet:
    def __init__(self, **kwargs):
        print("Initializing LLNet...")
        self.L1reg = kwargs.get('l1_strength', 0.01)
        self.learningRate = theano.shared(kwargs.get("learningRate", 0.001), name="learningRate")
        # self.x = T.matrix("x")
        self.x = T.tensor4("x")
        self.y = T.matrix("y")
        self.layers = []
        self.num_layers = 0
        print("COMPLETED: Intializing LLNet")

    def update_params(self):
        # Update parameter list
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        # Update Regularization
        self.regularization = 0.0
        for param in self.params:
            b = T.abs_(param)
            self.regularization += self.L1reg * b.sum()

    def update_graphs(self, supervised=1):
        # Update Gradients
        self.gparams = [T.grad(self.cost, param) for param in self.params]
        # Update Updates
        self.updates = [
                    (param, param - self.learningRate * gparam) for param, gparam in zip(self.params,self.gparams)
                    ]
        if supervised:
            self.train = theano.function(
                  inputs=[self.x,self.y],
                  outputs=self.cost,
                  updates=self.updates
            )
        else:
            self.train = theano.function(
                  inputs=[self.x],
                  outputs=self.cost,
                  updates=self.updates
            )

    def add_layer(self, input, dims, activator=None):
        self.layers += [HiddenLayer(input, dims, self.num_layers, activator)]
        self.num_layers += 1

    def add_conv_layer(self, input, filter_shape, image_shape, poolsize):
        self.layers += [convLayer.ConvLayer(input, filter_shape, image_shape, poolsize)]
        self.num_layers += 1

    def saveParams(self, file_name):
        # Get current date-time string
        name = strftime(file_name + "-" +"%Y-%m-%d_%H-%M-%S")
        with open(dirs.path + dirs.savedDataDirectory + name + dirs.savedDataExt, 'a+b') as out:
            params = {}
            for p in self.params:
                params[str(p)] = p.get_value()
            data = {'timeStamp' : name, 'params' : params}
            pickle.dump(data, out)
        print("COMPLETED: Saved parameters to {}".format(name + dirs.savedDataExt))
        return name

    def loadParams(self, name):
        lst = []
        with open(dirs.path + dirs.savedDataDirectory + name + dirs.savedDataExt, 'rb') as file:
            while 1:
                try:
                    lst.append(pickle.load(file))
                except EOFError:
                    break
        loadedParams = lst[0].get('params')
        for param in self.params:
            sParam = str(param)
            loadedValue = loadedParams.get(sParam)
            if not loadedValue.size:
                print("FATAL ERROR: Could not locate value of {}".format(sParam))
                sys.exit(0)
            else:
                param.set_value(loadedValue)
        print("COMPLETED: Loading parameters from {}".format(name))

