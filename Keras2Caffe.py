import keras
import numpy as np
from keras.models import Sequential, model_from_json
from keras import backend as K
import h5py
import caffe
from caffe import layers as L
from caffe import params as P
import caffe.proto.caffe_pb2 as pb2
import google.protobuf as pb

import six
from collections import OrderedDict, Counter

batch_size = 64
model_name = "idsia"

def convert_filter(numpy_filter_weight):
    return np.transpose(numpy_filter_weight,(3,2,1,0))

def convert_fc(numpy_fc_weight):
    return np.transpose(numpy_fc_weight,(1,0))

def keras_config_to_caffe_prototxt(keras_model):

    n = caffe.NetSpec()
    layers = keras_model.layers
    print([layer.name for layer in layers])
    last_layer = None
    relu_count = 1

    for idx, layer in enumerate(layers):
        if last_layer == None: # First layer
            batch_input_shape = layer.batch_input_shape

            # batch_input_shape = (None, 48, 48, 3)
            # n.data, n.label = L.DummyData(shape=[\
            #     dict(dim=[batch_size, batch_input_shape[3], batch_input_shape[1], \
            #         batch_input_shape[2]]), \
            #     dict(dim=[batch_size, 1, 1, 1])], \
            #     transform_param=dict(scale=1./255), ntop=2)

            n.data = L.Input(shape=dict(dim=[batch_size, batch_input_shape[3], batch_input_shape[1], batch_input_shape[2]]))
            last_layer = "data"

        if type(layer)==keras.layers.core.Flatten:
            continue

        layer_name = layer.name.split('_')[1]

        if type(layer)==keras.layers.convolutional.Conv2D:
            # param=caffe.Layer_param(layer.name,'Convolution')
            # net.add_layer_with_data(param,[w,b])

            # Add Conv and Relu layer
            setattr(n, layer_name, L.Convolution(getattr(n, last_layer), \
                kernel_size=layer.kernel_size[0], stride=layer.strides[0], num_output=layer.filters, \
                weight_filler=dict(type='gaussian', std=0.01)))
            last_layer = layer_name
            layer_name = "relu{}".format(relu_count)
            setattr(n, layer_name, L.ReLU(getattr(n, last_layer), in_place=True))
            relu_count += 1

            # Add parameters
            w, b=layer.get_weights()
            w =convert_filter(w)

        if type(layer) == keras.layers.convolutional.MaxPooling2D:
            setattr(n, layer_name, L.Pooling(getattr(n, last_layer), \
                kernel_size=layer.pool_size[0], stride=layer.strides[0], pool=P.Pooling.MAX))

        if type(layer) == keras.layers.core.Dense:
            # print(layer.units)
            setattr(n, layer_name, L.InnerProduct(getattr(n, last_layer), num_output=layer.units, \
                weight_filler=dict(type='gaussian', std=0.01)))
            last_layer = layer_name
            if layer.activation.__name__ == "relu":
                layer_name = "relu{}".format(relu_count)
                setattr(n, layer_name, L.ReLU(getattr(n, last_layer), in_place=True))
                relu_count += 1
            elif layer.activation.__name__ == "softmax":
                layer_name = "prob"
                setattr(n, layer_name, L.Softmax(getattr(n, last_layer)))
                # setattr(n, layer_name, L.SoftmaxWithLoss(getattr(n, last_layer), n.label))

        last_layer = layer_name

    with open("./converted_model/" + model_name + ".prototxt", 'w') as f:
        f.write(str(n.to_proto()))

def keras_weights_to_caffemodel(keras_model, caffe_net):
    for layer in keras_model.layers:

        if type(layer)==keras.layers.convolutional.Conv2D:
            layer_name = layer.name.split('_')[-1]

            # print(caffe_net.params[layer_name][0].data.shape)

            w, b = layer.get_weights()
            # print("xxxxx {} *****".format(layer_name))
            # print(w.shape)
            # print(w[:, :, 0, 0])
            w = convert_filter(w)
            # print("xxxxx")
            # print(w.shape)
            # print(w[0, 0, :, :])
            # print("xxxxxxxxxxxxxx")

            # net.params['conv1'][0] will be initialized, while net.layers[0].blobs[0] will not
            np.copyto(caffe_net.params[layer_name][0].data, w)
            np.copyto(caffe_net.params[layer_name][1].data, b)

        if type(layer) == keras.layers.core.Dense:
            layer_name = layer.name.split('_')[-1]

            # print(caffe_net.params[layer_name][0].data.shape)

            w, b = layer.get_weights()
            w = convert_fc(w)

            # net.params['conv1'][0] will be initialized, while net.layers[0].blobs[0] will not
            print(caffe_net.params[layer_name][0].data.shape)
            np.copyto(caffe_net.params[layer_name][0].data, w)
            print(caffe_net.params[layer_name][0].data.shape)
            print(caffe_net.params[layer_name][1].data.shape)
            np.copyto(caffe_net.params[layer_name][1].data, b)
            print(caffe_net.params[layer_name][1].data.shape)

    caffe_net.save("./converted_model/" + model_name + ".caffemodel")


def load_weights_from_hdf5(config_filename, weight_filename):

    js = open(config_filename, 'r').readlines()[0]

    keras_model = model_from_json(js)
    
    keras_model.load_weights(weight_filename)
    keras_model.summary()

    layers = keras_model.layers

    return keras_model

if __name__=='__main__':
    keras_model = load_weights_from_hdf5('./saved_models/keras_tensorflow_gpu_GT_config.json', \
        './saved_models/keras_tensorflow_gpu_GT_weights.hdf5')
    keras_config_to_caffe_prototxt(keras_model)
    caffe_net = caffe.Net("./converted_model/" + model_name + ".prototxt", caffe.TEST)
    keras_weights_to_caffemodel(keras_model, caffe_net)

    # Net with weights
    caffe_net = caffe.Net("./converted_model/" + model_name + ".prototxt", \
        "./converted_model/" + model_name + ".caffemodel", \
        caffe.TEST)
        
