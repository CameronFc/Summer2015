from llnet import LLNet
from format import Formatter
from header import dirs
from format import getDesiredFiles
import time
import sys
import numpy as np
import theano

# If we want to reverse the order w.x back to x.w, we need to remove the transposes in this file
# and also change the order of the layer parameters here and in LLNet.

#TODO: move transposes into LLNET

# Device to maximize interaction between LLNet and input data to absolutely maximize reusability
class MetaNet:
    def __init__(self, y_formatter, test_function, **options):
        self.train_file_limit = options.get('train_file_limit',50)
        self.test_file_limit = options.get('test_file_limit', 10)
        self.min_improvement = options.get('min_improvement', 0.01)

        self.file_name = options.get('file_name')
        self.test_name = options.get('test_name')
        self.anim_type = options.get('anim_type')
        self.supervised = options.get('supervised')
        self.l1_strength = options.get("l1_strength")
        self.y_formatter = y_formatter
        self.test_function = test_function
        self.input_dim = self.get_input_dim()
        self.test_output_layers = []
        self.learning_rate=options.get('learning_rate', 0.001)
        # Init net
        self.net = LLNet(learningRate=self.learning_rate, l1_strength=self.l1_strength)

    def add_layer(self, input,dims, activator=None):
        self.net.add_layer(input, dims, activator)
        self.net.update_params()

    def get_layer(self, index):
        return self.net.layers[index]

    def get_y(self):
        return self.net.y

    def get_x(self):
        return self.net.x

    def add_cost(self, func):
        self.net.cost = func + self.net.regularization
        self.net.update_graphs(supervised=self.supervised)

    def get_formatted_dataset(self, name, file_limit, type, anim_type):
        formatter = Formatter(anim_type=anim_type)
        set_x, set_y = formatter.get_dataset(type=type, name=name, file_limit=file_limit)
        for index, dict in enumerate(set_y):
            set_y[index] = self.format_y(dict)
        return set_x, set_y

    def format_y(self, dict):
        return self.y_formatter(dict)

    def get_input_dim(self):
        # Cheating: Get a single file from the training set, and see what size it is
        dataset_options = {
            'name': self.file_name,
            'type': "PTRAIN",
            'anim_type': self.anim_type,
            'file_limit': 1
        }
        set_x, set_y = self.get_formatted_dataset(**dataset_options)
        # Return the number of 'features' in the image
        return len(set_x[0])

    def train(self):
        # Get the formatted training set
        dataset_options = {
            'name': self.file_name,
            'type': "PTRAIN",
            'anim_type': self.anim_type,
            'file_limit': self.train_file_limit
        }
        train_set_x , train_set_y = self.get_formatted_dataset(**dataset_options)
        print("Beginning Training...")
        startTime = time.time()
        oldCost = sys.maxsize
        # 'While' loop with index
        for i in range(sys.maxsize):
            totalCost = 0
            for j in range(len(train_set_x)):
                x = np.array([train_set_x[j]]).transpose()
                y = train_set_y[j]
                cost = (
                    self.net.train(x,y) if self.supervised == 1
                    else self.net.train(x)
                )
                totalCost += cost
                #print("Cost for image {}: {}".format(j,cost))
            improvement = (oldCost - totalCost)/totalCost
            if(np.abs(improvement) < self.min_improvement):
                print("Reached minimum improvement threshold")
                break
            print("Trained {} steps, cost : {} improvement: +{:%}, learningRate: {}".format(str(i + 1),totalCost, improvement, self.net.learningRate.get_value()))
            oldCost = totalCost
        print("Completed Training in {} seconds".format(str(time.time() - startTime)))

    def test(self):
        dataset_options = {
            'name' : self.file_name,
            'type' : "PTESTS",
            'anim_type' : self.anim_type,
            'file_limit' : self.test_file_limit
        }
        test_x, test_y = self.get_formatted_dataset(**dataset_options)
        for index, element_x in enumerate(test_x):
            self.classify(test_x[index], test_y[index])

    def classify(self, image_x, image_y):
        get_test_output = theano.function(
            inputs=[self.net.x],
            outputs=[(self.get_layer(element).output) for element in self.test_output_layers]
        )
        # Why np.array([]).transpose()? need to go 'dim' -> [1,dim] -> [dim,1] as in training above
        # Outputs a 3-dimensional array of the outputs for each test layer specified.
        self.test_function(get_test_output(np.array([image_x]).transpose()), image_y)

    def save(self):
        name = self.test_name + "-" + self.file_name
        # Save as test-image-date
        self.net.saveParams(name)

    def load(self, file_name):
        self.net.loadParams(file_name)

    def load_last(self):
        path = dirs.path + dirs.savedDataDirectory
        files, names = getDesiredFiles(path, self.test_name)
        # Return the most recent one; should be in order of date
        self.net.loadParams(names[-1])


