from python import MetaNet as mn
import theano
import theano.tensor as T
from header import dirs

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
    'num_missed': 0
}

# Define how we want to display the results of testing
def test_function(value_arrays, image_y):
    # Value_arrays is always a 3-dimensional matrix
    print("Predicted size: ", value_arrays[2][0][0], "Actual size: ", image_y[2][0])
    print("Predicted class: ", value_arrays[1].argmax(), "Actual class: ", image_y[1][0])
    if value_arrays[1].argmax() - image_y[1][0]:
        iteratives['num_missed'] += 1


options = {
    'test_name': 'sizes',
    'file_name': "VaryingSizeCubes",
    'anim_type': "PANIM",
    'train_file_limit': 10,
    'test_file_limit': 10,
    'min_improvement': 0.01
}

Meta = mn.MetaNet(y_formatting_func, test_function, **options)

nfln = 10 # num first layer neurons
l2n = Meta.input_dim / 2
l3n = Meta.input_dim / 4
l4n = Meta.input_dim / 2

num_classes = 8
# 4 layers from 4x -> 2x-> x-> 2x-> 4x
Meta.add_layer(Meta.get_x(), [l2n, Meta.input_dim])
Meta.add_layer(Meta.get_layer(0).output, [l3n, l2n])
Meta.add_layer(Meta.get_layer(1).output, [l4n, l3n])
Meta.add_layer(Meta.get_layer(2).output, [Meta.input_dim, l4n])


# Add custom cost function
cost = ((Meta.get_layer(3).output - Meta.get_x())**2).sum()
# Regularization added automatically
Meta.add_cost(cost)

# Specify the indices of what layers we want to come out of the classify function
Meta.test_output_layers = [3]

Meta.train()
#Meta.test()
#Meta.save()
#Meta.load_last()
#Meta.test()
#print("Mis-classifications of colors: ", iteratives.get('num_missed'))