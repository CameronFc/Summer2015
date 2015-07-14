from llnet import LLNet
from format import Formatter
import time
import sys
import numpy as np
import theano
import theano.tensor as T

# Device to contain the training data and all the super-parameters of the net
class MetaNet:
    def __init__(self):
        self.train_file_limit = 10
        self.test_file_limit = 10
        self.file_names = "TestAnimation"
        self.anim_type = "PANIM"
        input_dim = self.get_input_dim()
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

    def get_formatted_dataset(self, name, file_limit, type, anim_type):
        formatter = Formatter(anim_type=anim_type)
        set_x, set_y = formatter.get_dataset(type=type, name=name, file_limit=file_limit)
        for index, dict in enumerate(set_y):
            #TODO: Change this when we alter scene again
            set_y[index] = [dict.get("objects")[0].get("light").get("position"),
                                       dict.get("objects")[1].get("rect").get("color")]
        return set_x, set_y

    def get_input_dim(self):
        # Cheating: Get a single file from the training set, and see what size it is
        dataset_options = {
            'name': self.file_names,
            'type': "PTRAIN",
            'anim_type': self.anim_type,
            'file_limit': 1
        }
        set_x, set_y = self.get_formatted_dataset(**dataset_options)
        # Return the number of 'features' in the image
        return len(set_x[0])

    def train(self, min_improvement=0.001):
        print("Beginning Training...")
        # Get the formatted training set
        dataset_options = {
            'name': self.file_names,
            'type': "PTRAIN",
            'anim_type': self.anim_type,
            'file_limit': self.train_file_limit
        }
        train_set_x , train_set_y = self.get_formatted_dataset(**dataset_options)
        startTime = time.time()
        #TODO: CHANGE ME
        oldCost = 10000
        # 'While' loop with index
        for i in range(sys.maxsize):
            totalCost = 0
            for j in range(len(train_set_x)):
                totalCost += self.net.train([train_set_x[j]], train_set_y[j])
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
        self.classify([train_set_x[4]])

    def test(self):
        dataset_options = {
            'name' : self.file_names,
            'type' : "PTESTS",
            'anim_type' : self.anim_type,
            'file_limit' : self.test_file_limit
        }
        test_x, test_y = self.get_formatted_dataset(**dataset_options)

    def classify(self, image):
        get_class = theano.function(
            inputs=[self.net.x],
            outputs=self.net.layers[2].output
        )
        print(get_class(image))

