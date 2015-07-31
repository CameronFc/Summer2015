import subprocess
import platform
import sys
from header import Dirs
import caffe
import lmdb
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

print "OS:     ", platform.platform()
print "Python: ", sys.version.split("\n")[0]
# No CUDA Yet
# print "CUDA:   ", subprocess.Popen(["nvcc","--version"], stdout=subprocess.PIPE).communicate()[0].split("\n")[3]
print "LMDB:   ", ".".join([str(i) for i in lmdb.version()])


df = pd.read_csv("train.csv", sep=",")
features = df.ix[:,1:-1].as_matrix()
labels = df.ix[:,-1].as_matrix()

vec_log = np.vectorize(lambda x: np.log(x+1))
vec_int = np.vectorize(lambda str: int(str[-1])-1)


features = vec_log(features)
labels = vec_int(labels)

# http://deepdish.io/2015/04/28/creating-lmdb-in-python/
def load_data_into_lmdb(lmdb_name, features, labels=None):
    path = Dirs.data
    env = lmdb.open(path + lmdb_name, map_size=features.nbytes*2)

    features = features[:,:,None,None]
    for i in range(features.shape[0]):
        datum = caffe.proto.caffe_pb2.Datum()

        datum.channels = features.shape[1]
        datum.height = 1
        datum.width = 1

        if features.dtype == np.int:
            datum.data = features[i].tostring()
        elif features.dtype == np.float:
            datum.float_data.extend(features[i].flat)
        else:
            raise Exception("features.dtype unknown.")

        if labels is not None:
            datum.label = int(labels[i])

        str_id = '{:08}'.format(i)
        with env.begin(write=True) as txn:
            txn.put(str_id, datum.SerializeToString())


load_data_into_lmdb("train_data_lmdb", features_training, labels_training)
load_data_into_lmdb("test_data_lmdb", features_testing, labels_testing)