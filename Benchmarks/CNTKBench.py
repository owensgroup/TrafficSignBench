import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
import time, sys, DLHelper

import cntk as C
from cntk.learners import momentum_sgd as SGD
from cntk import cross_entropy_with_softmax as Softmax, classification_error as ClassificationError
from cntk.io import MinibatchSourceFromData
from cntk.logging import ProgressPrinter
from cntk.train.training_session import *
from cntk.initializer import he_normal
from timeit import default_timer
import ResNet.cntk_resnet as cntk_resnet

class CNTKBench:
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
        # It's the Python wheel version instead of the setting in the script that decides which device (CPU/GPU) to use.
        self.devices = ['cpu']
        if C.device.use_default_device().type() == 0:
            print('running on CPU')
        else:
            print('running on GPU')
            self.devices = ['gpu']
        self.backends = args.backends

        self.cntk_model = None

        _ = DLHelper.create_dir(root, ["saved_data", "saved_models"], self.network_type, self.devices)

        print("**********************************")
        print("Training on CNTK")
        print("**********************************")

    def constructCNN(self, cntk_input):
        self.cntk_model = None
        if self.network_type == 'idsia':
            with C.layers.default_options(activation=C.relu):
                self.cntk_model = C.layers.Sequential([
                    C.layers.Convolution((3,3), strides=(1,1), num_filters=100, pad=False,
                        init=he_normal(), name="cntk_conv1"),
                    C.layers.MaxPooling((2,2), strides=(2,2), name="cntk_pool1"),
                    C.layers.Convolution((4,4), strides=(1,1), num_filters=150, pad=False,
                        init=he_normal(), name="cntk_conv2"),
                    C.layers.MaxPooling((2,2), strides=(2,2), name="cntk_pool2"),
                    C.layers.Convolution((3,3), strides=(1,1), num_filters=250, pad=False,
                        init=he_normal(), name="cntk_conv3"),
                    C.layers.MaxPooling((2,2), strides=(2,2), name="cntk_pool3"),

                    C.layers.Dense(200, init=he_normal(), name="cntk_fc1"),
                    C.layers.Dense(self.class_num, activation=None, init=he_normal(), name="cntk_fc2") # Leave the softmax for now
                ])(cntk_input)
        elif self.network_type == 'self':
            with C.layers.default_options(activation=C.relu):
                self.cntk_model = C.layers.Sequential([
                    C.layers.Convolution((5,5), strides=(2,2), num_filters=64, pad=True,
                        init=he_normal(), name="cntk_conv1"),
                    C.layers.MaxPooling((2,2), strides=(2,2), name="cntk_pool1"),
                    C.layers.Convolution((3,3), strides=(1,1), num_filters=256, pad=True,
                        init=he_normal(), name="cntk_conv2"),
                    C.layers.MaxPooling((2,2), strides=(2,2), name="cntk_pool2"),

                    C.layers.Dense(2048, init=he_normal(), name="cntk_fc1"),
                    C.layers.Dropout(0.5, name="cntk_dropout1"),
                    C.layers.Dense(self.class_num, activation=None, init=he_normal(), name="cntk_fc2") # Leave the softmax for now
                ])(cntk_input)
        elif self.network_type == "resnet-56":
            self.cntk_model = cntk_resnet.create_model(cntk_input, 9, self.class_num) # 6*9 + 2 = 56
        elif self.network_type == "resnet-32":
            self.cntk_model = cntk_resnet.create_model(cntk_input, 5, self.class_num) # 6*5 + 2 = 32
        elif self.network_type == "resnet-20":
            self.cntk_model = cntk_resnet.create_model(cntk_input, 3, self.class_num) # 6*3 + 2 = 20

    def benchmark(self):
        # Construct model, io and metrics
        cntk_input = C.input_variable((3, self.resize_size[0], self.resize_size[1]), np.float32)
        cntk_output = C.input_variable((self.class_num), np.float32)
        self.constructCNN(cntk_input)
        cntk_cost = Softmax(self.cntk_model, cntk_output)
        cntk_error = ClassificationError(self.cntk_model, cntk_output)

        # Construct data
        cntk_train_x = np.ascontiguousarray(np.vstack([np.expand_dims(x, axis=0).transpose([0,3,1,2]).astype('float32')/255 for x in self.x_train]), dtype=np.float32)
        cntk_valid_x = np.ascontiguousarray(np.vstack([np.expand_dims(x, axis=0).transpose([0,3,1,2]).astype('float32')/255 for x in self.x_valid]), dtype=np.float32)
        cntk_test_x = np.ascontiguousarray(np.vstack([np.expand_dims(x, axis=0).transpose([0,3,1,2]).astype('float32')/255 for x in self.testImages]), dtype=np.float32)

        cntk_train_y = C.one_hot(C.input_variable(1), self.class_num, sparse_output=False)(np.expand_dims(np.array(self.y_train, dtype='f'), axis=1))
        cntk_valid_y = C.one_hot(C.input_variable(1), self.class_num, sparse_output=False)(np.expand_dims(np.array(self.y_valid, dtype='f'), axis=1))
        cntk_test_y = C.one_hot(C.input_variable(1), self.class_num, sparse_output=False)(np.expand_dims(np.array(self.testLabels, dtype='f'), axis=1))


        # Trainer and mb source
        cntk_learner = SGD(self.cntk_model.parameters, lr=0.01, momentum=0.9, unit_gain=False, use_mean_gradient=True) # To compare performance with other frameworks
        cntk_trainer = C.Trainer(self.cntk_model, (cntk_cost, cntk_error), cntk_learner)
        cntk_train_src = C.io.MinibatchSourceFromData(dict(x=C.Value(cntk_train_x), y=C.Value(cntk_train_y)), max_samples=len(cntk_train_x))
        cntk_valid_src = C.io.MinibatchSourceFromData(dict(x=C.Value(cntk_valid_x), y=C.Value(cntk_valid_y)), max_samples=len(cntk_valid_x))
        cntk_test_src = C.io.MinibatchSourceFromData(dict(x=C.Value(cntk_test_x), y=C.Value(cntk_test_y)), max_samples=len(cntk_test_x))

        # Mapping for training, validation and testing
        def getMap(src, bs):
            batch = src.next_minibatch(bs)
            return {
                cntk_input: batch[src.streams['x']],
                cntk_output: batch[src.streams['y']]
            }

        # Create log file
        train_batch_count = len(self.x_train) // self.batch_size + 1
        valid_batch_count = len(self.x_valid) // self.batch_size + 1
        test_batch_count = len(self.testImages) // self.batch_size + 1
        filename = "{}saved_data/{}/{}/callback_data_cntk_{}_{}by{}_{}.h5".format(self.root, self.network_type, self.devices[0], self.dataset, self.resize_size[0], self.resize_size[1], self.preprocessing)
        f = DLHelper.init_h5py(filename, self.epoch_num, train_batch_count * self.epoch_num)

        # Start training
        try:
            batch_count = 0
            f['.']['time']['train']['start_time'][0] = time.time()

            # Each epoch
            for epoch in range(0, self.epoch_num):
                cntk_train_src.restore_from_checkpoint({'cursor': 0, 'total_num_samples': 0})
                cntk_valid_src.restore_from_checkpoint({'cursor': 0, 'total_num_samples': 0})
                cntk_test_src.restore_from_checkpoint({'cursor': 0, 'total_num_samples': 0})

                # Each batch
                for i in range(train_batch_count):
                    batch_count += 1

                    # Read a mini batch from the training data file
                    data = getMap(cntk_train_src, self.batch_size)

                    # Train a batch
                    start = default_timer()
                    cntk_trainer.train_minibatch(data)

                    # Save training loss
                    training_loss = cntk_trainer.previous_minibatch_loss_average

                    # Save batch time. Prevent asynchronous
                    train_batch_time = default_timer() - start
                    f['.']['time']['train_batch'][batch_count-1] = train_batch_time

                    # Continue saving training loss
                    eval_error = cntk_trainer.previous_minibatch_evaluation_average
                    f['.']['cost']['train'][batch_count-1] = np.float32(training_loss)
                    f['.']['accuracy']['train'][batch_count-1] = np.float32((1.0 - eval_error) * 100.0)

                    if i % 30 == 0: # Print per 50 batches
                        print("Epoch: {0}, Minibatch: {1}, Loss: {2:.4f}, Error: {3:.2f}%".format(epoch, i, training_loss, eval_error * 100.0))

                # Save batch marker
                f['.']['time_markers']['minibatch'][epoch] = np.float32(batch_count)

                # Validation
                validation_loss = 0
                validation_error = 0
                for j in range(valid_batch_count):
                    # Read a mini batch from the validation data file
                    data = getMap(cntk_valid_src, self.batch_size)

                    # Valid a batch
                    batch_x, batch_y = data[cntk_input].asarray(), data[cntk_output].asarray()
                    validation_loss += cntk_cost(batch_x, batch_y).sum()
                    validation_error += cntk_trainer.test_minibatch(data) * len(batch_x)

                validation_loss /= len(self.x_valid)
                validation_error /= len(self.x_valid)

                # Save validation loss for the whole epoch
                f['.']['cost']['loss'][epoch] = np.float32(validation_loss)
                f['.']['accuracy']['valid'][epoch] = np.float32((1.0 - validation_error) * 100.0)
                print("[Validation]")
                print("Epoch: {0}, Loss: {1:.4f}, Error: {2:.2f}%\n".format(epoch, validation_loss, validation_error * 100.0))

            # Save related params
            f['.']['time']['train']['end_time'][0] = time.time() # Save training time
            f['.']['config'].attrs["total_minibatches"] = batch_count
            f['.']['time_markers'].attrs['minibatches_complete'] = batch_count

            # Testing
            test_error = 0
            for j in range(test_batch_count):
                # Read a mini batch from the validation data file
                data = getMap(cntk_test_src, self.batch_size)

                # Valid a batch
                test_error += cntk_trainer.test_minibatch(data) * data[cntk_input].num_samples

            test_error /= len(self.testImages)

            f['.']['infer_acc']['accuracy'][0] = np.float32((1.0 - test_error) * 100.0)
            print("Accuracy score is %f" % (1.0 - test_error))

            self.cntk_model.save("{}saved_models/{}/{}/cntk_{}_{}by{}_{}.pth".format(self.root, self.network_type, self.devices[0], self.dataset, self.resize_size[0], self.resize_size[1], self.preprocessing))

        except KeyboardInterrupt:
            pass
        except Exception as e:
            raise e
        finally:
            print("Close file descriptor")
            f.close()



# # Validation and testing configuration
# cntk_valid_config = CrossValidationConfig(
#     minibatch_source = cntk_valid_src,
#     frequency = (1, DataUnit.sweep),
#     minibatch_size = batch_size,
#     model_inputs_to_streams = valid_map,
#     max_samples = len(x_valid),
#     criterion = (cntk_cost, cntk_error))
# cntk_test_config = TestConfig(
#     minibatch_source = cntk_test_src,
#     minibatch_size = batch_size,
#     model_inputs_to_streams = test_map,
#     criterion = (cntk_cost, cntk_error))

# # Start training
# training_session(
#         trainer = cntk_trainer,
#         mb_source = cntk_train_src,
#         mb_size = batch_size,
#         model_inputs_to_streams = train_map,
#         max_samples = len(x_train) * epoch_num,
#         progress_frequency = len(x_train),
#         cv_config = cntk_valid_config,
#         test_config = cntk_test_config).train()
