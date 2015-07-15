#
# Template file for the construction of custom neural nets
#

import MetaNet as mn
import theano
import theano.tensor as T
from header import dirs

dirs.path += "../"

def y_formatting_func(dict):
    return [
            dict.get("objects")[0].get("light").get("position"),
            [dict.get("objects")[1].get("rect").get("color_class"),0,0]
            ] # Need vectors of same length in matrix


options = {
    'file_names' : "TestAnimation",
    'anim_type': "PANIM",
    'train_file_limit': 50,
    'test_file_limit': 10,
    'min_improvement': 0.01
}

# Create net handler
Meta = mn.MetaNet(y_formatting_func, **options)

# Add Layers
nfln = 10 # num first layer neurons
num_classes = 8
Meta.add_layer(Meta.get_x(), [nfln, Meta.input_dim], T.tanh)
Meta.add_layer(Meta.get_layer(0).output, [3, nfln])
# Add an n-sized multi-layer to be handled by a softmax
Meta.add_layer(Meta.get_layer(0).output, [num_classes,nfln])

# Add custom cost function
position_cost = ((Meta.get_layer(1).output.transpose() - Meta.get_y()[0])**2).sum()
exp_out = T.exp(Meta.get_layer(2).output)
normalized = (exp_out / exp_out.sum())
class_index = theano.tensor.cast(Meta.get_y()[1][0], 'int64')
correct_class_prob = normalized[class_index]
color_cost =  -T.log(correct_class_prob)[0] # Returns an array; use first and only value
Meta.add_cost(position_cost + color_cost * 30)

Meta.train()
Meta.test()