import caffe
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


caffe.set_mode_cpu()

def load_net(net_name, model_name):
    return caffe.Net("../net_sources/" + net_name + ".prototxt",
                    "../net_sources/model_states/" + model_name + ".caffemodel",
                    caffe.TEST)

net = load_net("custom_autoencoder", "custom_autoencoder_iter_100")

print net.blobs['data'].data.shape

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape, 'decn1': net.blobs['decn1'].data.shape})
transformer.set_transpose('data', (2,0,1))
# transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
# transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'))
# out = net.forward()
# print("Predicted class is #{}.".format(out['prob'].argmax()))
#
# plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))

#.transpose(0,2,3,1).reshape(10*60,80,3)
img = "../images/pvr_images/CaffeImage12D.jpg"
#src = net.blobs['data']
#src.data[...] = misc.imread(img).transpose(2,0,1)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img))
#print src.data.mean()
print misc.imread(img).transpose(2,0,1).shape
# out = net.forward()
print net.blobs['data'].data[0].sum()
misc.imsave("autoencoder_output_debug.jpg", transformer.deprocess('data', net.blobs['data'].data[0]))

image = misc.imread(img)

print(type(net.blobs['data'].data[0][0,0,0]))
print(type(image[0,0,0].astype('uint8')))
print([x for x in net._layer_names])
print(net.blobs['decn1'].data.shape)

im = misc.imread("autoencoder_output.jpg")

print "im mean is: " , im.mean()

print im.shape

#out_flat = net.forward(start='data', end='flatdata')

out = net.forward(end='decn1')


# print "flatdata mean: ", out_flat['flatdata'].mean()
# print "flatdata max: ", out_flat['flatdata'].max()
# print "flatdata min: ", out_flat['flatdata'].min()

print "other flatdata max:", net.blobs['flatdata'].data.max()

print out['decn1'].mean()

img_out = out['decn1']

# transformer.deprocess('decn1', out['decn1'])
img_out *= 255
img_out = img_out.astype('uint8')
print img_out.mean()
print img_out.max()
print img_out.min()

#print out[1].min()
misc.imsave("autoencoder_output.jpg", img_out.reshape(20,3,60,80)[0])

#print out
# Images are exactly the same, must be matplotlib that is in error
#plt.imshow(net.blobs['data'].data[0].transpose(1,2,0), cmap=plt.cm.gray)
#plt.imshow(net.blobs['data'].data[0].astype('uint8').transpose(1,2,0), cmap=plt.cm.gray)
#plt.show()
# plt.imshow(misc.imread(img), cmap=plt.cm.gray)
# plt.show()

#plt.imshow(misc.imread(img).transpose(2,0,1).transpose(1,2,0), cmap=plt.cm.gray)
#plt.show()
