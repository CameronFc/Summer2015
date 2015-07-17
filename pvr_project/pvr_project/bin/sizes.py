import MetaNet as mn
from header import dirs
import theano
import theano.tensor as T

class CustomFunctions:
    # Define how dicts of scenes are processed
    def __init__(self):
        self.iteratives = {
            'num_missed': 0
        }

    def y_formatting_func(self, dict):
        return [
                dict.get("objects")[0].get("light").get("position"),
                [dict.get("objects")[1].get("rect").get("color_class"),0,0],
                #TODO: CHANGE COLON AFTER SIZE IF WE RE-RENDER! #mistake!
                [dict.get("objects")[1].get("rect").get("size:"),0,0]
                ] # Need vectors of same length in matrix
    
    # Define how we want to display the results of testing
    def test_function(self, value_arrays, image_y):
        # Value_arrays is always a 3-dimensional matrix
        print("Predicted size: ", value_arrays[2][0][0], "Actual size: ", image_y[2][0])
        print("Predicted class: ", value_arrays[1].argmax(), "Actual class: ", image_y[1][0])
        if value_arrays[1].argmax() - image_y[1][0]:
            self.iteratives['num_missed'] += 1

class Sizes:
    def __init__(self):
        self.CF = CustomFunctions()
        self.options = {
            'test_name': 'sizes',
            'file_name': "VaryingSizeCubes",
            'anim_type': "PANIM",
            'supervised': 1,
            'train_file_limit': 50,
            'test_file_limit': 10,
            'min_improvement': 0.001
        }
        self.y_formatting_func = self.CF.y_formatting_func
        self.test_func = self.CF.test_function
        self.create_meta_net()
        
    def create_meta_net(self):
        self.Meta = mn.MetaNet(self.y_formatting_func, self.test_func, **self.options)
        nfln = 40 # num first layer neurons
        num_classes = 8
        # Base non-linear layer
        self.Meta.add_layer(self.Meta.get_x(), [nfln, self.Meta.input_dim], T.tanh)
        # Position estimation layer
        self.Meta.add_layer(self.Meta.get_layer(0).output, [3, nfln])
        # Add an n-sized multi-layer to be handled by a softmax # Color layer
        self.Meta.add_layer(self.Meta.get_layer(0).output, [num_classes,nfln])
        # Size estimator layer
        self.Meta.add_layer(self.Meta.get_layer(0).output, [1, nfln])
        
        # Add custom cost function
        position_cost = ((self.Meta.get_layer(1).output.transpose() - self.Meta.get_y()[0])**2).sum()
        size_cost = (self.Meta.get_layer(3).output[0][0] - self.Meta.get_y()[2][0])**2
        # Color cost section
        exp_out = T.exp(self.Meta.get_layer(2).output)
        normalized = (exp_out / exp_out.sum())
        class_index = theano.tensor.cast(self.Meta.get_y()[1][0], 'int64')
        correct_class_prob = normalized[class_index]
        color_cost =  -T.log(correct_class_prob)[0] # Returns an array; use first and only value
        # Regularization added automatically
        self.Meta.add_cost(color_cost)

        # Specify the indices of what layers we want to come out of the classify function
        self.Meta.test_output_layers = [1,2,3]

    def train(self):
        self.Meta.train()

    def test(self):
        self.Meta.test()
        print("Mis-classifications of colors: ", self.CF.iteratives.get('num_missed'))

    def save(self):
        self.Meta.save()

    def load_last(self):
        self.Meta.load_last()
