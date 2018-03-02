import sys
import caffe
import numpy as np
import cv2
import DLHelper
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing import image

def load_weights_from_hdf5(config_filename, weight_filename):

    js = open(config_filename, 'r').readlines()[0]

    keras_model = model_from_json(js)
    
    keras_model.load_weights(weight_filename)
    keras_model.summary()

    layers = keras_model.layers

    return keras_model

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off'); plt.show()

# root = "/Users/moderato/Documents/Libraries/caffe/models/"
# size = (227, 227)

# pimga = cv2.imread(root + "cat.jpg")
# pimga = cv2.resize(pimga, size)
# nimga = np.array(pimga).reshape(1,size[0],size[1],3).transpose(0,3,1,2)

# net = caffe.Net(root + "bvlc_reference_caffenet/deploy.prototxt", root + "bvlc_reference_caffenet.caffemodel", caffe.TEST)


root = "/Users/moderato/Downloads/"
resize_size = (48, 48)
dataset = "GT"

root, trainImages, trainLabels, testImages, testLabels, class_num = DLHelper.getImageSets(root, resize_size, dataset=dataset, printing=False)

keras_model = load_weights_from_hdf5('./saved_models/keras_tensorflow_gpu_GT_config.json', \
        './saved_models/keras_tensorflow_gpu_GT_weights.hdf5')

img = cv2.imread("./00000.ppm")
# [6, 5, 48, 49]
img = img[5:49, 6:48] / 255.0
img = cv2.resize(img, resize_size)

# keras_predictions = np.argmax(keras_model.predict(np.expand_dims(img, axis=0)))

keras_test_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in testImages])

keras_predictions = [np.argmax(keras_model.predict(np.expand_dims(feature, axis=0))) for feature in keras_test_x[0:5]]

# [16, 1, 38, 33, 11, 38, 18, 12, 25, 35]
print(keras_predictions)

from keras import backend as K

inp = keras_model.input                                           # input placeholder
outputs = [layer.output for layer in keras_model.layers]          # all layer outputs
functor = K.function([inp]+ [K.learning_phase()], outputs) # evaluation function

# Testing
test = np.expand_dims(img, axis=0)
layer_outs = functor([test, 1.])
# print(test)
print(layer_outs[1])
print(layer_outs[1].shape)

# first_layer = keras_model.layers[0]
# w, b = first_layer.get_weights()
# print(b)
# print(b.shape)




net = caffe.Net("./converted_model/idsia.prototxt", "./converted_model/idsia.caffemodel", caffe.TEST)

# # nimga = np.array(testImages[0])
# # shape = nimga.shape
nimga = np.array(img).reshape(1, resize_size[0], resize_size[1], 3).transpose(0, 3, 1, 2)


# nimgas = [np.array(img).reshape(1, resize_size[0], resize_size[1], 3).transpose(0, 3, 1, 2) / 255.0 for img in testImages[1:20]]
# out = net.predict(nimgas)
out = net.forward_all(**{"data": nimga})

print(net.blobs['conv1'].data)
print(net.blobs['conv1'].data.shape)
# print(net.blobs[''])
# print(net.blobs['data'].data)

# # # print(out)
# # print(testLabels[1:20])
# for o in out:
print("Predicted class is #{}.".format(out['prob'].argmax()))