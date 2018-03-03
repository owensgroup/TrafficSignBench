import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
import time, sys, DLHelper

from neon.backends import gen_backend, cleanup_backend
from neon.initializers import (
    Gaussian, 
    Constant, 
    Kaiming
)
from neon.layers import GeneralizedCost, Affine
from neon.backends.backend import Block
from neon.layers import (
    Conv as Conv, 
    Dropout, 
    Pooling
)
from neon.transforms import (
    Rectlin,
    Softmax, 
    CrossEntropyMulti, 
    Misclassification, 
    TopKMisclassification, 
    Accuracy
)
from neon.models import Model
from neon.optimizers import (
    GradientDescentMomentum as SGD, 
    RMSProp, 
    ExpSchedule
)
from neon.callbacks.callbacks import Callbacks, Callback, LossCallback
from neon.data.dataiterator import ArrayIterator
from timeit import default_timer
from ResNet.neon_resnet import resnet

# This callback class is actually a mix of LossCallback and MetricCallback
class SelfCallback(LossCallback):
    def __init__(self, eval_set, test_set, epoch_freq):
        super(SelfCallback, self).__init__(eval_set=eval_set, epoch_freq=epoch_freq)
        self.train_batch_time = None
        self.total_batch_index = 0
        self.test_set = test_set
        self.metric = Accuracy()
        
    def on_train_begin(self, callback_data, model, epochs):
        super(SelfCallback, self).on_train_begin(callback_data, model, epochs)
        
        # Save training time per batch
        total_batches = callback_data['config'].attrs['total_minibatches']
        tb = callback_data.create_dataset("time/train_batch", (total_batches,))
        tb.attrs['time_markers'] = 'minibatch'

        acc = callback_data.create_group('accuracy')
        acc_v = acc.create_dataset('valid', (epochs,))
        acc_v.attrs['time_markers'] = 'epoch_freq'
        acc_v.attrs['epoch_freq'] = 1
        acc_t = acc.create_dataset('train', (total_batches,))
        acc_t.attrs['time_markers'] = 'minibatch'

        acc = callback_data.create_group('infer_acc')
        acc_v = acc.create_dataset('accuracy', (1,))

    def on_epoch_end(self, callback_data, model, epoch):
        super(SelfCallback, self).on_epoch_end(callback_data, model, epoch)
        callback_data['accuracy/valid'][epoch] = model.eval(self.eval_set, metric=Accuracy())[0] * 100.0

    def on_minibatch_begin(self, callback_data, model, epoch, minibatch):
        self.train_batch_time = default_timer()

    def on_minibatch_end(self, callback_data, model, epoch, minibatch):
        callback_data["time/train_batch"][self.total_batch_index] = (default_timer() - self.train_batch_time)
        self.total_batch_index += 1

    def on_train_end(self, callback_data, model):
        callback_data['infer_acc/accuracy'][0] = model.eval(self.test_set, metric=Accuracy())[0] * 100.0

class SelfModel(Model):
    def __init__(self, layers, dataset=None, weights_only=False, name="model", optimizer=None):
        super(SelfModel, self).__init__(layers=layers, dataset=dataset, weights_only=weights_only, name=name, optimizer=optimizer)

    def _epoch_fit(self, dataset, callbacks):
        epoch = self.epoch_index
        self.total_cost[:] = 0
        # iterate through minibatches of the dataset
        for mb_idx, (x, t) in enumerate(dataset):
            callbacks.on_minibatch_begin(epoch, mb_idx)
            self.be.begin(Block.minibatch, mb_idx)

            x = self.fprop(x)

            # Save per-minibatch accuracy
            acc = Accuracy()
            mbstart = callbacks.callback_data['time_markers/minibatch'][epoch - 1] if epoch > 0 else 0
            callbacks.callback_data['accuracy/train'][mbstart + mb_idx] = acc(x, t) * 100.0

            self.total_cost[:] = self.total_cost + self.cost.get_cost(x, t)

            # deltas back propagate through layers
            # for every layer in reverse except the 0th one
            delta = self.cost.get_errors(x, t)

            self.bprop(delta)
            self.optimizer.optimize(self.layers_to_optimize, epoch=epoch)

            self.be.end(Block.minibatch, mb_idx)
            callbacks.on_minibatch_end(epoch, mb_idx)

        # now we divide total cost by the number of batches,
        # so it was never total cost, but sum of averages
        # across all the minibatches we trained on
        self.total_cost[:] = self.total_cost / dataset.nbatches

class NeonBench:
    def __init__(self, args, root, x_train, x_valid, y_train, y_valid, testImages, testLabels, class_num):

        self.root = root
        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.testImages = testImages
        self.testLabels = testLabels
        self.class_num = class_num

        self.network_type = args.network_type
        self.resize_size = (args.resize_side, args.resize_side)
        self.dataset = args.dataset
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        self.preprocessing = args.preprocessing
        self.printing = args.printing
        self.devices = args.devices
        self.backends = args.backends

        self.neon_model = None

        _ = DLHelper.create_dir(root, ["saved_data", "saved_models"], self.network_type, self.devices)

        print("**********************************")
        print("Training on Neon")
        print("**********************************")

    def constructCNN(self):
        layers = []
        if self.network_type == "idsia":
            layers.append(Conv((3, 3, 100), strides=1, init=Kaiming(), bias=Constant(0.0), activation=Rectlin(), name="Conv1"))
            layers.append(Pooling(2, op="max", strides=2, name="neon_pool1"))
            layers.append(Conv((4, 4, 150), strides=1, init=Kaiming(), bias=Constant(0.0), activation=Rectlin(), name="Conv2"))
            layers.append(Pooling(2, op="max", strides=2, name="neon_pool2"))
            layers.append(Conv((3, 3, 250), strides=1, init=Kaiming(), bias=Constant(0.0), activation=Rectlin(), name="Conv3"))
            layers.append(Pooling(2, op="max", strides=2, name="neon_pool3"))
            layers.append(Affine(nout=200, init=Kaiming(local=False), bias=Constant(0.0), activation=Rectlin(), name="neon_fc1"))
            layers.append(Affine(nout=self.class_num, init=Kaiming(local=False), bias=Constant(0.0), activation=Softmax(), name="neon_fc2"))
        elif self.network_type == "self":
            layers.append(Conv((5, 5, 64), strides=2, padding=2, init=Kaiming(), bias=Constant(0.0), activation=Rectlin(), name="Conv1"))
            layers.append(Pooling(2, op="max", strides=2, name="neon_pool1"))
            layers.append(Conv((3, 3, 512), strides=1, padding=1, init=Kaiming(), bias=Constant(0.0), activation=Rectlin(), name="Conv2"))
            layers.append(Pooling(2, op="max", strides=2, name="neon_pool2"))
        #     layers.append(Pooling(5, op="avg", name="neon_global_pool"))
            layers.append(Affine(nout=2048, init=Kaiming(local=False), bias=Constant(0.0), activation=Rectlin(), name="neon_fc1"))
            layers.append(neon_Dropout(keep=0.5, name="neon_dropout1"))
            layers.append(Affine(nout=self.class_num, init=Kaiming(local=False), bias=Constant(0.0), activation=Softmax(), name="neon_fc2"))
        elif self.network_type == "resnet-56":
            layers = resnet(9, self.class_num, int(resize_size[0]/4)) # 6*9 + 2 = 56
        elif self.network_type == "resnet-32":
            layers = resnet(5, self.class_num, int(resize_size[0]/4)) # 6*5 + 2 = 32
        elif self.network_type == "resnet-20":
            layers = resnet(3, self.class_num, int(resize_size[0]/4)) # 6*3 + 2 = 20
        return layers

    def benchmark(self):
        for d in self.devices:
            b = d if (self.backends is None) or ("mkl" not in self.backends) else "mkl"
            print("Use {} as backend.".format(b))

            # Set up backend
            # backend: 'cpu' for single cpu, 'mkl' for cpu using mkl library, and 'gpu' for gpu
            be = gen_backend(backend=b, batch_size=self.batch_size, rng_seed=542, datatype=np.float32)

            # Make iterators
            neon_train_set = ArrayIterator(X=np.asarray([t.flatten().astype('float32')/255 for t in self.x_train]), y=np.asarray(self.y_train), make_onehot=True, nclass=self.class_num, lshape=(3, self.resize_size[0], self.resize_size[1]))
            neon_valid_set = ArrayIterator(X=np.asarray([t.flatten().astype('float32')/255 for t in self.x_valid]), y=np.asarray(self.y_valid), make_onehot=True, nclass=self.class_num, lshape=(3, self.resize_size[0], self.resize_size[1]))
            neon_test_set = ArrayIterator(X=np.asarray([t.flatten().astype('float32')/255 for t in self.testImages]), y=np.asarray(self.testLabels), make_onehot=True, nclass=self.class_num, lshape=(3, self.resize_size[0], self.resize_size[1]))

            # Initialize model object
            self.neon_model = SelfModel(layers=self.constructCNN())

            # Costs
            neon_cost = GeneralizedCost(costfunc=CrossEntropyMulti())

            # Model summary
            self.neon_model.initialize(neon_train_set, neon_cost)
            print(self.neon_model)

            # Learning rules
            neon_optimizer = SGD(0.01, momentum_coef=0.9, schedule=ExpSchedule(0.2))
            # neon_optimizer = RMSProp(learning_rate=0.0001, decay_rate=0.95)

            # # Benchmark for 20 minibatches
            # d[b] = self.neon_model.benchmark(neon_train_set, cost=neon_cost, optimizer=neon_optimizer)

            # Reset model
            # self.neon_model = None
            # self.neon_model = Model(layers=layers)
            # self.neon_model.initialize(neon_train_set, neon_cost)

            # Callbacks: validate on validation set
            callbacks = Callbacks(self.neon_model, eval_set=neon_valid_set, metric=Misclassification(3), output_file="{}saved_data/{}/{}/callback_data_neon_{}_{}_{}by{}_{}.h5".format(self.root, self.network_type, d, b, self.dataset, self.resize_size[0], self.resize_size[1], self.preprocessing))
            callbacks.add_callback(SelfCallback(eval_set=neon_valid_set, test_set=neon_test_set, epoch_freq=1))

            # Fit
            start = time.time()
            self.neon_model.fit(neon_train_set, optimizer=neon_optimizer, num_epochs=self.epoch_num, cost=neon_cost, callbacks=callbacks)
            print("Neon training finishes in {:.2f} seconds.".format(time.time() - start))

            # Result
            # results = self.neon_model.get_outputs(neon_valid_set)

            # Print error on validation set
            start = time.time()
            neon_error_mis = self.neon_model.eval(neon_valid_set, metric=Misclassification())*100
            print('Misclassification error = {:.1f}%. Finished in {:.2f} seconds.'.format(neon_error_mis[0], time.time() - start))

            # start = time.time()
            # neon_error_top3 = self.neon_model.eval(neon_valid_set, metric=TopKMisclassification(3))*100
            # print('Top 3 Misclassification error = {:.1f}%. Finished in {:.2f} seconds.'.format(neon_error_top3[2], time.time() - start))

            # start = time.time()
            # neon_error_top5 = self.neon_model.eval(neon_valid_set, metric=TopKMisclassification(5))*100
            # print('Top 5 Misclassification error = {:.1f}%. Finished in {:.2f} seconds.'.format(neon_error_top5[2], time.time() - start))

            self.neon_model.save_params("{}saved_models/{}/{}/neon_weights_{}_{}_{}by{}_{}.prm".format(self.root, self.network_type, d, b, self.dataset, self.resize_size[0], self.resize_size[1], self.preprocessing))

            # Print error on test set
            start = time.time()
            neon_error_mis_t = self.neon_model.eval(neon_test_set, metric=Misclassification())*100
            print('Misclassification error = {:.1f}% on test set. Finished in {:.2f} seconds.'.format(neon_error_mis_t[0], time.time() - start))

            # start = time.time()
            # neon_error_top3_t = self.neon_model.eval(neon_test_set, metric=TopKMisclassification(3))*100
            # print('Top 3 Misclassification error = {:.1f}% on test set. Finished in {:.2f} seconds.'.format(neon_error_top3_t[2], time.time() - start))

            # start = time.time()
            # neon_error_top5_t = self.neon_model.eval(neon_test_set, metric=TopKMisclassification(5))*100
            # print('Top 5 Misclassification error = {:.1f}% on test set. Finished in {:.2f} seconds.'.format(neon_error_top5_t[2], time.time() - start))

            cleanup_backend()
            self.neon_model = None
