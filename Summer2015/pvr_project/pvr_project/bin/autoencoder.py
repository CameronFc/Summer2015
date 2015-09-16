import MetaNet as MN
from format import displayImage
from format import readImage
from header import dirs

dirs.path += "../"

class CustomFunctions():
    # Define how dicts of scenes are processed
    @staticmethod
    def y_formatting_func(dict):
        return [
            dict.get("objects")[0].get("light").get("position"),
            [dict.get("objects")[1].get("rect").get("color_class"), 0, 0],
            # TODO: CHANGE COLON AFTER SIZE IF WE RE-RENDER! #mistake!
            [dict.get("objects")[1].get("rect").get("size:"), 0, 0]
        ]  # Need vectors of same length in matrix

    # Define how we want to display the results of testing
    @staticmethod
    def test_function(value_arrays, image_y):
        # Shape of array: (1,14400,1)
        # Value_arrays is always a 3-dimensional matrix
        displayImage(value_arrays[0].reshape((60, 80, 3)))

class AutoEncoder():
    def __init__(self):
        self.options = {
            'test_name': 'auto',
            'file_name': "StaticCube",
            'anim_type': "PSTATIC",
            'supervised': 0,
            'flatten_input': False,
            'train_file_limit': 100,
            'test_file_limit': 10,
            'learning_rate': 0.01,
            'min_improvement': 0.001,
            'l1_strength': 0
        }
        self.y_formatting_func = CustomFunctions.y_formatting_func
        self.test_func = CustomFunctions.test_function
        self.create_meta_net()

    def create_meta_net(self):
        self.Meta = MN.MetaNet(self.y_formatting_func, self.test_func, **self.options)
        # Specify layer dimensions
        l2n = 1000
        l3n = 500
        l4n = l2n
        num_classes = 8
        # 4 layers from 4x -> 2x-> x-> 2x-> 4x
        # self.Meta.add_conv_layer(input=self.Meta.get_x(), filter_shape=(1,3,5,5), image_shape=(1,3,60,80), poolsize=(2,2))
        self.Meta.add_layer(input=self.Meta.get_x(), dims=[3,80,60])
        # self.Meta.add_layer(input=self.Meta.get_layer(0).output, dims=[3,60*80])
        # self.Meta.add_layer(input=self.Meta.get_layer(1).output, dims=[3,60*80])
        # Add custom cost function
        cost = (self.Meta.get_layer(0).output * 0).sum()
        # cost = (self.Meta.get_layer(2).output - self.Meta.get_x()**2).sum()
        # Regularization added automatically
        self.Meta.add_cost(cost)
        # Specify the indices of what layers we want to come out of the classify function
        self.Meta.test_output_layers = [0]

    def train(self):
        self.Meta.train()

    def test(self):
        self.Meta.test()

    def save(self):
        self.Meta.save()

    def load_last(self):
        self.Meta.load_last()

aut = AutoEncoder()
# aut.train()
aut.test()