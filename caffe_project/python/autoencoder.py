from __future__ import print_function
import caffe
import os
from caffe import layers as L, params as P
from header import Dirs

from google.protobuf import text_format
from caffe.draw import get_pydot_graph
from caffe.proto import caffe_pb2

def encoder_layer(bottom, num_out):
    return L.InnerProduct(bottom,
                         # Global learningrate and decayrate multipliers for this layer
                         # For the parameters and bias respectively
                         param=[dict(lr_mult=1,decay_mult=1),
                                dict(lr_mult=1,decay_mult=0)],
                         inner_product_param=dict(
                             num_output =num_out,
                             weight_filler=dict(
                                 type='gaussian',
                                 std=1,
                                 sparse=15
                             ),
                             bias_filler=dict(
                                 type='constant',
                                 value=0
                             ))
                         )

# Define the structure of the autoencoder here
def caffenet(train_lmdb, test_lmdb, batch_size=10):
    n = caffe.NetSpec()
    # Define data layers
    n.data = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=train_lmdb,
                    # phase == 'TRAIN'
                    # include=[dict(phase=0)],
                    transform_param=dict(scale=1./255), ntop=1)
    # n.data_test = L.Data(name="data", batch_size=batch_size, backend=P.Data.LMDB, source=test_lmdb,
    #                 # phase == 'TEST'
    #                 include=[dict(phase=1)],
    #                 transform_param=dict(scale=1./255), ntop=1)

    # Stack of Innerproduct->sigmoid layers
    n.enc1 = encoder_layer(n.data, 1000)
    n.encn1 = L.Sigmoid(n.enc1)
    n.enc2 = encoder_layer(n.encn1, 500)
    n.encn2 = L.Sigmoid(n.enc2)
    n.enc3 = encoder_layer(n.encn2, 250)
    n.encn3 = L.Sigmoid(n.enc3)
    n.enc4 = encoder_layer(n.encn3, 30)
    n.dec4 = encoder_layer(n.enc4, 250)
    n.decn4 = L.Sigmoid(n.dec4)
    n.dec3 = encoder_layer(n.decn4, 500)
    n.decn3 = L.Sigmoid(n.dec3)
    n.dec2 = encoder_layer(n.decn3, 1000)
    n.decn2 = L.Sigmoid(n.dec2)
    n.dec1 = encoder_layer(n.decn2, 784)
    n.decn1 = L.Sigmoid(n.dec1)

    # Flatten the data so it can be compared to the output of the stack
    n.flatdata = L.Flatten(n.data)

    # Loss layers
    n.cross_entropy_loss = L.SigmoidCrossEntropyLoss(n.dec1, n.flatdata)
    n.euclidean_loss = L.EuclideanLoss(n.decn1, n.flatdata)
    return n.to_proto()

def make_net():
    with open(Dirs.core_path + 'custom_autoencoder.prototxt', 'w') as f:
        f.write("name: \"Custom_Autoencoder\"\n")
        print(caffenet('../net_sources/mnist_train_lmdb', "../net_sources/mnist_test_lmdb"), file=f)

def make_graph():
    try:
        os.remove("custom_graph.png")
    except:
        pass
    _net = caffe_pb2.NetParameter()
    f = open("../net_sources/custom_autoencoder.prototxt")
    text_format.Merge(f.read(), _net)
    get_pydot_graph(_net,"TB").write_png("custom_graph.png")

make_net()
make_graph()





