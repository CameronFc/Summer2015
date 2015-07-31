from __future__ import print_function
from caffe import layers as L, params as P, to_proto
import caffe
import matplotlib.pyplot as plt
import lmdb

from google.protobuf import text_format
from caffe.draw import get_pydot_graph
from caffe.proto import caffe_pb2
import scipy.misc as misc
import pydot

from header import Dirs

def caffenet(lmdb, batch_size=10):
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()


def make_net():
    with open(Dirs.core_path + 'train.prototxt', 'w') as f:
        print(caffenet('../net_sources/lmdb_train'), file=f)

    with open(Dirs.core_path + 'test.prototxt', 'w') as f:
        print(caffenet('../net_sources/lmdb_test'), file=f)

    # with open(Dirs.core_path + 'test.prototxt', 'w') as f:
    #     print(caffenet('/path/to/caffe-val-lmdb', batch_size=50, include_acc=True), file=f)

# if __name__ == '__main__':
#     make_net()

# make_net()
#
# solver = caffe.SGDSolver('../net_sources/lenet_solver.prototxt')
# #print(list((k, v.data.shape) for k, v in solver.net.blobs.items()))
#
# solver.net.forward()
# print(solver.test_nets[0].forward())
# print(solver.net.blobs['data'].data[1].shape)
# plt.imshow(solver.net.blobs['data'].data[:10].transpose(0,2,3,1).reshape(10*60,80,3), cmap=plt.cm.gray)
# print(solver.net.blobs['label'].data[:8])
# plt.show()

_net = caffe_pb2.NetParameter()
f = open("../net_sources/mnist_autoencoder.prototxt")
text_format.Merge(f.read(), _net)
get_pydot_graph(_net,"TB").write_png("graph.png")

