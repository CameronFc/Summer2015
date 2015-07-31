import MetaNet as MN
from format import displayImage


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

class TestEncoder():
    def __init__(self):
        self.options = {
            'test_name': 'auto',
            'file_name': "StaticCube",
            'anim_type': "PSTATIC",
            # Does the cost function involve (y) in any way?
            'supervised': 0,
            'train_file_limit': 100,
            'test_file_limit': 10,
            'learning_rate': 0.01,
            'min_improvement': 0.01,
            'l1_strength': 1
        }
        self.y_formatting_func = CustomFunctions.y_formatting_func
        self.test_func = CustomFunctions.test_function
        self.create_meta_net()

    def create_meta_net(self):
        self.Meta = MN.MetaNet(self.y_formatting_func, self.test_func, **self.options)
        # Specify layer dimensions
        l2n = 10000
        num_classes = 8
        # 4 layers from 4x -> 2x-> x-> 2x-> 4x
        self.Meta.add_layer(self.Meta.get_x(), [l2n, self.Meta.input_dim])
        # Add custom cost function
        cost = ((self.Meta.get_layer(0).output - self.Meta.get_x()) ** 2).sum() * 0.1 ** 20
        # Regularization added automatically
        self.Meta.add_cost(cost)
        # Specify the indices of what layers we want to come out of the classify function
        self.Meta.test_output_layers = [1]

    def train(self):
        self.Meta.train()

    def test(self):
        self.Meta.test()

    def save(self):
        self.Meta.save()

    def load_last(self):
        self.Meta.load_last()
