from python import MetaNet as mn
from header import dirs
from format import displayImage

dirs.path += "../"

# Define how dicts of scenes are processed
def y_formatting_func(dict):
    return [
            dict.get("objects")[0].get("light").get("position"),
            [dict.get("objects")[1].get("rect").get("color_class"),0,0],
            #TODO: CHANGE COLON AFTER SIZE IF WE RE-RENDER! #mistake!
            [dict.get("objects")[1].get("rect").get("size:"),0,0]
            ] # Need vectors of same length in matrix

iteratives = {

}

# Define how we want to display the results of testing
def test_function(value_arrays, image_y):
    # Shape of array: (1,14400,1)
    # Value_arrays is always a 3-dimensional matrix
    print(value_arrays[0].transpose())
    displayImage(value_arrays[0].reshape((60,80, 3)))

options = {
    'test_name': 'auto',
    'file_name': "StaticCube",
    'anim_type': "PSTATIC",
    'supervised': 0,
    'train_file_limit': 10,
    'test_file_limit': 10,
    'learning_rate': (0.001),
    'min_improvement': 0.01,
    'l1_strength': 1
}

Meta = mn.MetaNet(y_formatting_func, test_function, **options)

# Meta.input_dim / 2
l2n = 400
l3n = 50
l4n = l2n

num_classes = 8
# 4 layers from 4x -> 2x-> x-> 2x-> 4x
Meta.add_layer(Meta.get_x(), [l2n, Meta.input_dim])
Meta.add_layer(Meta.get_layer(0).output, [l3n, l2n])
Meta.add_layer(Meta.get_layer(1).output, [l4n, l3n])
Meta.add_layer(Meta.get_layer(2).output, [Meta.input_dim, l4n])


# Add custom cost function
cost = ((Meta.get_layer(3).output - Meta.get_x())**2).sum() * 0.1**20
# Regularization added automatically
Meta.add_cost(cost)

# Specify the indices of what layers we want to come out of the classify function
Meta.test_output_layers = [3]

Meta.train()
Meta.save()
#Meta.load_last()
#Meta.test()
