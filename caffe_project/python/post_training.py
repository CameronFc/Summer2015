import caffe
import matplotlib.pyplot as plt
from scipy import misc


caffe.set_mode_cpu()

def load_net(net_name, model_name):
    return caffe.Net("../net_sources/" + net_name + ".prototxt",
                    "../net_sources/model_states/" + model_name + ".caffemodel",
                    caffe.TEST)

net = load_net("custom_autoencoder", "custom_autoencoder_iter_300")

print net.blobs['data'].data.shape

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2,0,1))
# transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
# transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
# transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
#
# net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'))
# out = net.forward()
# print("Predicted class is #{}.".format(out['prob'].argmax()))
#
# plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))

#.transpose(0,2,3,1).reshape(10*60,80,3)
img = "../images/CaffeImage15D.jpg"
net.blobs['data'].data[0] = misc.imread(img).transpose(2,0,1)
print misc.imread(img).transpose(2,0,1).shape
# out = net.forward()
print net.blobs['data'].data[0].sum()
image = misc.imread(img)


#print out
#plt.imshow(net.blobs['data'].data[0].transpose(1,2,0), cmap=plt.cm.gray)
plt.imshow(net.blobs['data'].data[0].transpose(1,2,0), cmap=plt.cm.gray)
plt.show()
# plt.imshow(misc.imread(img), cmap=plt.cm.gray)
# plt.show()

#plt.imshow(misc.imread(img).transpose(2,0,1).transpose(1,2,0), cmap=plt.cm.gray)
#plt.show()
