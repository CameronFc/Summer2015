import numpy as np
import theano
import theano.tensor as T
import math
import sys
import pickle
import time
from time import strftime
from header import dirs
from format import Formatter
rng = np.random

#Logistic-linear net to estimate the position of lights
#TODO: Add batching function

#theano.config.compute_test_value = 'warn'

class HiddenLayer:
    def __init__(self, input, dims, activation=None, layer=1):
        wBase = rng.randn(*dims)
        self.w = theano.shared(wBase, name='w' + str(layer))
        self.b = theano.shared(0. , name='b' + str(layer))
        linOutput = T.dot(input, self.w) + self.b
        self.output = (
            linOutput if activation is None
            else activation(linOutput)
        )
        self.params = [self.w,self.b]

class LLNet:
    def __init__(self, **kwargs):
        print("Initializing LLNet...")
        self.L1reg = 0.0001
        self.learningRate = theano.shared(kwargs.get("learningRate", 0.001), name="learningRate")
        self.x = T.matrix("x")
        self.y = T.matrix("y")
        self.layers = []
        print("COMPLETED: Intializing LLNet")

    def update_params(self):
        # Update parameter list
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        # Update Regularization
        self.regularization = 0.0
        for i in (param.sum() for param in self.params):
            self.regularization += self.L1reg * i

    def update_graphs(self):
        # Update Gradients
        self.gparams = [T.grad(self.cost, param) for param in self.params]
        # Update Updates
        self.updates = [
                    (param, param - self.learningRate * gparam) for param, gparam in zip(self.params,self.gparams)
                    ]
        self.train = theano.function(
              inputs=[self.x,self.y],
              outputs=self.cost,
              updates=self.updates
        )

    def add_layer(self, input, dims, activator=None):
        self.layers += [HiddenLayer(input, dims, activator)]

    def saveParams(self):
        name = strftime("%Y-%m-%d_%H-%M-%S")
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
        #print(lst)
        loadedParams = lst[0].get('params')
        #For each parameter in the LLNet
        for param in self.params:
            sParam = str(param)
            #print(loadedParams.get(sParam))
            loadedValue = loadedParams.get(sParam)
            #print(type(loadedValue))
            if not loadedValue.size:
                print("FATAL ERROR: Could not locate value of {}".format(sParam))
                sys.exit(0)
            else:
                param.set_value(loadedValue)
        print("COMPLETED: Loading parameters from {}".format(name))

