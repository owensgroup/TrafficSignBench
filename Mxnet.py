import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
import time, sys, DLHelper

print("**********************************")
print("Training on Mxnet")
print("**********************************")

if sys.platform == "darwin":
    root = "/Users/moderato/Downloads/"
else:
    root = "/home/zhongyilin/Desktop/"
print(root)

network_type = sys.argv[1]
if network_type == "idsia":
    resize_size = (48, 48)
else:
    resize_size = (int(sys.argv[2]), int(sys.argv[2]))
dataset = sys.argv[3]
epoch_num = int(sys.argv[4])
batch_size = int(sys.argv[5])
process = sys.argv[6]
printing = True if sys.argv[7] == '1' else False
backends = sys.argv[8:]
print("Training on {}".format(backends))

root, trainImages, trainLabels, testImages, testLabels, class_num = DLHelper.getImageSets(root, resize_size, dataset=dataset, process=process, printing=printing)
x_train, x_valid, y_train, y_valid = ms.train_test_split(trainImages, trainLabels, test_size=0.2, random_state=542)

_ = DLHelper.create_dir(root, ["saved_data", "saved_models"], network_type, backends)

import mxnet as mx
import logging
from timeit import default_timer
from mxnet_resnet import get_symbol
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

# class MxCustomInit(mx.initializer.Initializer):
#     def __init__(self, idict):
#         super(MxCustomInit, self).__init__()
#         self.dict = idict
#         np.random.seed(seed=1)

#     def _init_weight(self, name, arr):
#         if name in self.dict.keys():
#             dictPara = self.dict[name]
#             for(k, v) in dictPara.items():
#                 arr = np.random.normal(0, v, size=arr.shape)

#     def _init_bias(self, name, arr):
#         if name in self.dict.keys():
#             dictPara = self.dict[name]
#             for(k, v) in dictPara.items():
#                 arr[:] = v

class MxBatchCallback(object):
    def __init__(self):
        pass

    def __call__(self, param):
        pass

        # param.epoch: Epoch index
        # param.eval_metric: Real time metrics (Use param.eval_metric.get_name_value() to get it)
        # param.nbatch: Current batch count (in one epoch)
        # param.locals: Miscellaneous

def constructCNN(cnn_type='self'):
    mx_softmax = None
    if cnn_type == 'idsia':
        data = mx.sym.Variable('data')
        mx_conv1 = mx.sym.Convolution(data = data, name='mx_conv1', num_filter=100, kernel=(3,3), stride=(1,1))
        mx_act1 = mx.sym.Activation(data = mx_conv1, name='mx_relu1', act_type="relu")
        mx_mp1 = mx.sym.Pooling(data = mx_act1, name = 'mx_pool1', kernel=(2,2), stride=(2,2), pool_type='max')
        mx_conv2 = mx.sym.Convolution(data = mx_mp1, name='mx_conv2', num_filter=150, kernel=(4,4), stride=(1,1))
        mx_act2 = mx.sym.Activation(data = mx_conv2, name='mx_relu2', act_type="relu")
        mx_mp2 = mx.sym.Pooling(data = mx_act2, name = 'mx_pool2', kernel=(2,2), stride=(2,2), pool_type='max')
        mx_conv3 = mx.sym.Convolution(data = mx_mp2, name='mx_conv3', num_filter=250, kernel=(3,3), stride=(1,1))
        mx_act3 = mx.sym.Activation(data = mx_conv3, name='mx_relu3', act_type="relu")
        mx_mp3 = mx.sym.Pooling(data = mx_act3, name = 'mx_pool3', kernel=(2,2), stride=(2,2), pool_type='max')
        mx_fl = mx.sym.Flatten(data = mx_mp3, name="mx_flatten")
        mx_fc1 = mx.sym.FullyConnected(data = mx_fl, name='mx_fc1', num_hidden=200)
        mx_fc2 = mx.sym.FullyConnected(data = mx_fc1, name='mx_fc2', num_hidden=class_num)
        mx_softmax = mx.sym.SoftmaxOutput(data = mx_fc2, name ='softmax')
    elif cnn_type == 'self':
        data = mx.sym.Variable('data')
        mx_conv1 = mx.sym.Convolution(data = data, name='mx_conv1', num_filter=64, kernel=(5,5), stride=(2,2), pad=(2,2))
        mx_act1 = mx.sym.Activation(data = mx_conv1, name='mx_relu1', act_type="relu")
        mx_mp1 = mx.sym.Pooling(data = mx_act1, name = 'mx_pool1', kernel=(2,2), stride=(2,2), pool_type='max')
        mx_conv2 = mx.sym.Convolution(data = mx_mp1, name='mx_conv2', num_filter=512, kernel=(3,3), stride=(1,1), pad=(1,1))
        mx_act2 = mx.sym.Activation(data = mx_conv2, name='mx_relu2', act_type="relu")
        mx_mp2 = mx.sym.Pooling(data = mx_act2, name = 'mx_pool2', kernel=(2,2), stride=(2,2), pool_type='max')
        mx_fl = mx.sym.Flatten(data = mx_mp2, name="mx_flatten")
        mx_fc1 = mx.sym.FullyConnected(data = mx_fl, name='mx_fc1', num_hidden=2048)
        mx_drop = mx.sym.Dropout(data = mx_fc1, name='mx_dropout1', p=0.5)
        mx_fc2 = mx.sym.FullyConnected(data = mx_drop, name='mx_fc2', num_hidden=class_num)
        mx_softmax = mx.sym.SoftmaxOutput(data = mx_fc2, name ='softmax')
    elif cnn_type == 'resnet-56':
        mx_softmax = get_symbol(class_num, 56, "{},{},{}".format(3, resize_size[0], resize_size[1]))
    elif cnn_type == 'resnet-32':
        mx_softmax = get_symbol(class_num, 32, "{},{},{}".format(3, resize_size[0], resize_size[1]))
    elif cnn_type == 'resnet-20':
        mx_softmax = get_symbol(class_num, 20, "{},{},{}".format(3, resize_size[0], resize_size[1]))

    return mx_softmax

# Prepare image sets
# batch size = (batch, 3, size_x, size_y)
mx_train_x = mx.nd.array([i.swapaxes(0,2).astype("float32")/255 for i in x_train])
mx_valid_x = mx.nd.array([i.swapaxes(0,2).astype("float32")/255 for i in x_valid])
mx_test_x = mx.nd.array([i.swapaxes(0,2).astype("float32")/255 for i in testImages])
mx_train_y = mx.nd.array(y_train, dtype=np.float32) # No need of one_hot
mx_valid_y = mx.nd.array(y_valid, dtype=np.float32)
mx_test_y = mx.nd.array(testLabels, dtype=np.float32)

# The iterators have input name of 'data' and output name of 'softmax_label' if not particularly specified
mx_train_set = mx.io.NDArrayIter(mx_train_x, mx_train_y, batch_size, shuffle=True)
mx_valid_set = mx.io.NDArrayIter(mx_valid_x, mx_valid_y, batch_size)
mx_test_set = mx.io.NDArrayIter(mx_test_x, mx_test_y, batch_size)

# Print the shape and type of training set lapel
# mx_train_set.provide_label

mx_softmax = constructCNN(network_type)

# Print the names of arguments in the model
# mx_softmax.list_arguments() # Make sure the input and the output names are consistent of those in the iterator!!

# Print the size of the model
# mx_softmax.infer_shape(data=(1,3,49,49))

# Draw the network
# mx.viz.plot_network(mx_softmax, shape={"data":(batch_size, 3, resize_size[0], resize_size[1])})

# Initialization params
# mx_nor_dict = {'normal': 0.01}
# mx_cons_dict = {'constant': 0.0}
# mx_init_dict = {}
# for layer in mx_softmax.list_arguments():
#     hh = layer.split('_')
#     if hh[-1] == 'weight':
#         mx_init_dict[layer] = mx_nor_dict
#     elif hh[-1] == 'bias':
#         mx_init_dict[layer] = mx_cons_dict
# print(mx_init_dict)

for b in backends:
    print("Using {} backend".format(b))
    # create a trainable module on CPU/GPU
    mx_model =None
    if b == 'cpu':
        mx_model = mx.mod.Module(context = mx.cpu(), symbol = mx_softmax)
    else: # GPU
        mx_model = mx.mod.Module(context = mx.gpu(0), symbol = mx_softmax)

    max_total_batch = (len(x_train) // batch_size + 1) * epoch_num
    filename = "{}/saved_data/{}/{}/callback_data_mxnet_{}_{}by{}_{}.h5".format(root, network_type, b, dataset, resize_size[0], resize_size[0], process)
    f = DLHelper.init_h5py(filename, epoch_num, max_total_batch)

    try:
        # # Train the model
        # # Currently no solution to reproducibility. Eyes on issue 47.

        # allocate memory given the input data and label shapes
        mx_model.bind(data_shapes=mx_train_set.provide_data, label_shapes=mx_train_set.provide_label)
        # initialize parameters by he-normal random numbers
        mx_model.init_params(mx.initializer.MSRAPrelu('in', 0.0))
        # use SGD with learning rate 0.1 to train
        mx_model.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.01), ('momentum', 0.9)))
        # use accuracy as the metric
        acc = mx.metric.create('acc')
        mx_metric = mx.metric.create([acc, 'ce'])
        # train 5 epochs, i.e. going over the data iter one pass

        batch_count = 0
        f['.']['time']['train']['start_time'][0] = time.time()
        for epoch in range(epoch_num):
            mx_train_set.reset()
            mx_valid_set.reset()
            mx_metric.reset()

            epoch_batch = 0 # The batch index in this epoch
            epoch_start = default_timer()
            for batch in mx_train_set:
                start = default_timer()
                batch_count += 1
                epoch_batch += 1
                mx_model.forward(batch, is_train=True)       # compute predictions
                mx_model.update_metric(mx_metric, batch.label)  # accumulate prediction accuracy
                mx_model.backward()                          # compute gradients
                mx_model.update()                            # update parameters

                # Save batch time
                train_batch_time = default_timer() - start
                f['.']['time']['train_batch'][batch_count-1] = train_batch_time

                # Save training loss
                f['.']['cost']['train'][batch_count-1] = np.float32(mx_metric.get_name_value()[1][1])
                f['.']['accuracy']['train'][batch_count-1] = np.float32(mx_metric.get_name_value()[0][1] * 100.0)
                print("Epoch: {}, batch: {}, accuracy: {:.3f}, loss: {:.6f}, batch time: {:.3f}s"\
                    .format(epoch, epoch_batch-1, mx_metric.get_name_value()[0][1],\
                        mx_metric.get_name_value()[1][1], train_batch_time))
            
            # Save batch marker
            f['.']['time_markers']['minibatch'][epoch] = np.float32(batch_count)
            print('Epoch {}, Training {}, Time {:.2f}s'.format(epoch, mx_metric.get_name_value(), default_timer()-epoch_start))

            mx_metric.reset()
            for batch in mx_valid_set:
                mx_model.forward(batch, is_train=False)      # validation instead of training
                mx_model.update_metric(mx_metric, batch.label)
                # No need for backward or update

            # Save validation loss for the whole epoch
            f['.']['cost']['loss'][epoch] = np.float32(mx_metric.get_name_value()[1][1])
            f['.']['accuracy']['valid'][epoch] = np.float32(mx_metric.get_name_value()[0][1] * 100.0)

            print('Epoch %d, Validation %s' % (epoch, mx_metric.get_name_value()))

        # Save related params
        f['.']['time']['train']['end_time'][0] = time.time()
        f['.']['config'].attrs["total_minibatches"] = batch_count
        f['.']['time_markers'].attrs['minibatches_complete'] = batch_count

        score = mx_model.score(mx_test_set, ['acc'])
        f['.']['infer_acc']['accuracy'][0] = np.float32(score[0][1] * 100.0)
        print("Accuracy score is %f" % (score[0][1]))

        mx_model.save_params("{}saved_models/{}/{}/mxnet_{}_{}by{}_{}.params".format(root, network_type, b, dataset, resize_size[0], resize_size[0], process))
        mx_model._symbol.save("{}saved_models/{}/{}/mxnet_{}_{}by{}_{}.json".format(root, network_type, b, dataset, resize_size[0], resize_size[0], process))
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e
    finally:
        print("Close file descriptor")
        f.close()
