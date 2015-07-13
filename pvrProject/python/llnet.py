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

# Device to contain the training data and all the super-parameters of the net
class MetaNet:
    def __init__(self):
        formatter = Formatter(animType="PANIM")
        file_limit = 10
        # Get the training dataset
        self.train_set_x, self.train_set_y = formatter.get_dataset(type="PTRAIN", name="TestAnimation", file_limit=file_limit)
        for index, dict in enumerate(self.train_set_y):
            #TODO: Change this when we alter scene again
            self.train_set_y[index] = [dict.get("objects")[0].get("light").get("position"),
                                       dict.get("objects")[1].get("rect").get("color")]
        input_dim = len(self.train_set_x[0])
        # Init net
        self.net = LLNet()
        # Add layers
        self.net.add_layer(self.net.x, [input_dim, 10], T.tanh)
        self.net.add_layer(self.net.layers[0].output, [10, 3])
        self.net.add_layer(self.net.layers[0].output, [10, 3])
        # Update parameters to produce regularization weight
        self.net.update_params()
        # Add custom cost function
        position_cost = ((self.net.layers[1].output - self.net.y[0])**2).sum()
        color_cost = ((self.net.layers[2].output - self.net.y[1])**2).sum()
        self.net.cost = position_cost + color_cost * 10 + self.net.regularization
        # Compile internal theano functions
        self.net.update_graphs()

    def train(self, min_improvement=0.001):
        print("Beginning Training...")
        startTime = time.time()
        #TODO: CHANGE ME
        oldCost = 10000
        # 'While' loop with index
        for i in range(sys.maxsize):
            totalCost = 0
            for j in range(len(self.train_set_x)):
                totalCost += self.net.train([self.train_set_x[j]], self.train_set_y[j])
            #if i + 1 == 1 or i + 1 == steps or (i + 1) % math.ceil(math.sqrt(steps)) == 0:

            improvement = (oldCost - totalCost)/totalCost
            #new_learning_rate = self.learningRate.get_value() * math.exp(improvement)
            #self.learningRate.set_value(new_learning_rate)
            if(np.abs(improvement) < min_improvement):
                print("Reached minimum improvement threshold")
                break
            print("Trained {} steps, cost : {} improvement: +{:%}, learningRate: {}".format(str(i + 1),totalCost, improvement, self.net.learningRate.get_value()))
            oldCost = totalCost
        print("Completed Training in {} seconds".format(str(time.time() - startTime)))
        #DEBUG
        self.classify([self.train_set_x[4]])
        print(self.train_set_y[4][1])

    def classify(self, image):
        get_class = theano.function(
            inputs=[self.net.x],
            outputs=self.net.layers[2].output
        )
        print(get_class(image))

