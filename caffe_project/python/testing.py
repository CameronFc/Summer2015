import caffe
from scipy import misc
import sys

sys.path.append("../net_sources/")

solver = caffe.SGDSolver('../net_sources/custom_autoencoder_solver.prototxt')
solver.net.copy_from('../net_sources/model_states/custom_autoencoder_iter_100.caffemodel')

transformer = caffe.io.Transformer({'data': solver.net.blobs['data'].data.shape, 'decn1': solver.net.blobs['decn1'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
img = "../images/pvr_images/CaffeImage19D.jpg"
solver.net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))

solver.step(4)

out = solver.net.blobs['euclidean_loss'].data

print dir(solver.net)
print solver.net.outputs


print out