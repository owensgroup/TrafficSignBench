import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
import time, sys, DLHelper

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import model_from_json
from timeit import default_timer
from importlib import reload
import tensorflow as tf
import os
import ResNet.keras_resnet

# Function to dynamically change keras backend
def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

class LossHistory(Callback):
    def __init__(self, filename, epoch_num, max_total_batch):
        super(Callback, self).__init__()
        
        self.batch_count = 0
        self.epoch_num = epoch_num
        self.filename = filename
        self.batch_time = None
        self.max_total_batch = max_total_batch
        self.f = DLHelper.init_h5py(filename, epoch_num, max_total_batch)
    
    def on_train_begin(self, logs={}):
        try:
            self.f['.']['time']['train']['start_time'][0] = default_timer()
        except Exception as e:
            self.f.close()
            raise e

    def on_epoch_end(self, epoch, logs={}):
        try:
            print(logs)
            self.f['.']['cost']['loss'][epoch] = np.float32(logs.get('val_loss'))
            self.f['.']['accuracy']['valid'][epoch] = np.float32(logs.get('val_acc') * 100.0)
            self.f['.']['time_markers']['minibatch'][epoch] = np.float32(self.batch_count)
        except Exception as e:
            self.f.close()
            raise e
        
    def on_batch_begin(self, batch, logs={}):
        try:
            self.batch_time = default_timer()
        except Exception as e:
            self.f.close()
            raise e
    
    def on_batch_end(self, batch, logs={}):
        try:
            self.f['.']['cost']['train'][self.batch_count] = np.float32(logs.get('loss'))
            self.f['.']['accuracy']['train'][self.batch_count-1] = np.float32(logs.get('acc') * 100.0)
            self.f['.']['time']['train_batch'][self.batch_count] = (default_timer() - self.batch_time)
            self.batch_count += 1
        except Exception as e:
            self.f.close()
            raise e
        
    def on_train_end(self, logs=None):
        try:
            self.f['.']['time']['train']['end_time'][0] = default_timer()
            self.f['.']['config'].attrs["total_minibatches"] = self.batch_count
            self.f['.']['time_markers'].attrs['minibatches_complete'] = self.batch_count
        except Exception as e:
            self.f.close()
            raise e

class KerasBench:
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

        self.keras_model = None

        _ = DLHelper.create_dir(root, ["saved_data", "saved_models"], self.network_type, self.devices)

        print("**********************************")
        print("Training on Keras")
        print("**********************************")

        print("Training on {}".format(self.backends))

    def constructCNN(self):
        self.keras_model = Sequential()
        if self.network_type == "idsia":
            self.keras_model.add(Conv2D(100, (3, 3), strides=(1, 1), activation="relu", input_shape=(self.resize_size[0], self.resize_size[1], 3), 
                kernel_initializer='he_normal', bias_initializer='zeros', name="keras_conv1"))
            self.keras_model.add(MaxPooling2D(pool_size=(2, 2), name="keras_pool1"))
            self.keras_model.add(Conv2D(150, (4, 4), strides=(1, 1), activation="relu", 
                kernel_initializer='he_normal', bias_initializer='zeros', name="keras_conv2"))
            self.keras_model.add(MaxPooling2D(pool_size=(2, 2), name="keras_pool2"))
            self.keras_model.add(Conv2D(250, (3, 3), strides=(1, 1), activation="relu", 
                kernel_initializer='he_normal', bias_initializer='zeros', name="keras_conv3"))
            self.keras_model.add(MaxPooling2D(pool_size=(2, 2), name="keras_pool3"))
            self.keras_model.add(Flatten(name="keras_flatten")) # An extra layer to flatten the previous layer in order to connect to fully connected layer
            self.keras_model.add(Dense(200, activation="relu", 
                kernel_initializer='he_normal', bias_initializer='zeros', name="keras_fc1"))
            self.keras_model.add(Dense(self.class_num, activation="softmax", 
                kernel_initializer='he_normal', bias_initializer='zeros', name="keras_fc2"))
        elif self.network_type == "self":
            self.keras_model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation="relu", input_shape=(self.resize_size[0], self.resize_size[1], 3), 
                kernel_initializer='he_normal', bias_initializer='zeros', name="keras_conv1"))
            self.keras_model.add(MaxPooling2D(pool_size=(2, 2), name="keras_pool1"))
            self.keras_model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation="relu", 
                kernel_initializer='he_normal', bias_initializer='zeros', name="keras_conv2"))
            self.keras_model.add(MaxPooling2D(pool_size=(2, 2), name="keras_pool2"))
            self.keras_model.add(Flatten(name="keras_flatten")) # An extra layer to flatten the previous layer in order to connect to fully connected layer
            self.keras_model.add(Dense(2048, activation="relu", 
                kernel_initializer='he_normal', bias_initializer='zeros', name="keras_fc1"))
            self.keras_model.add(Dropout(0.5, name="keras_dropout1"))
            self.keras_model.add(Dense(self.class_num, activation="softmax", 
                kernel_initializer='he_normal', bias_initializer='zeros', name="keras_fc2"))
        elif self.network_type == "resnet-56":
            self.keras_model = keras_resnet.resnet_v1((self.resize_size[0], self.resize_size[1], 3), 50, num_classes=self.class_num)
        elif self.network_type == "resnet-32":
            self.keras_model = keras_resnet.resnet_v1((self.resize_size[0], self.resize_size[1], 3), 32, num_classes=self.class_num)
        elif self.network_type == "resnet-20":
            self.keras_model = keras_resnet.resnet_v1((self.resize_size[0], self.resize_size[1], 3), 20, num_classes=self.class_num)

    def benchmark(self):
        # Load and process images
        keras_train_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in self.x_train]).astype('float32')
        keras_valid_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in self.x_valid]).astype('float32')
        keras_test_x = np.vstack([np.expand_dims(image.img_to_array(x), axis=0).astype('float32')/255 for x in self.testImages]).astype('float32')
        keras_train_y = to_categorical(self.y_train, self.class_num)
        keras_valid_y = to_categorical(self.y_valid, self.class_num)
        keras_test_y = to_categorical(self.testLabels, self.class_num)

        for b in self.backends:
            for device in self.devices:
                set_keras_backend(b)
                max_total_batch = (len(self.x_train) // self.batch_size + 1) * self.epoch_num

                # Build model
                self.constructCNN()

                # self.keras_model.summary()
                keras_optimizer = SGD(lr=0.01, decay=1.6e-8, momentum=0.9) # Equivalent to decay rate 0.2 per epoch? Need to re-verify
                # keras_optimizer = RMSProp(lr=0.01, decay=0.95)
                keras_cost = "categorical_crossentropy"
                self.keras_model.compile(loss=keras_cost, optimizer=keras_optimizer, metrics=["acc"])

                checkpointer = ModelCheckpoint(filepath="{}saved_models/{}/{}/keras_{}_{}_{}by{}_{}_weights.hdf5".format(self.root, self.network_type, device, b, self.dataset, self.resize_size[0], self.resize_size[1], self.preprocessing),
                                                   verbose=1, save_best_only=True)
                losses = LossHistory("{}saved_data/{}/{}/callback_data_keras_{}_{}_{}by{}_{}.h5".format(self.root, self.network_type, device, b, self.dataset, self.resize_size[0], self.resize_size[1], self.preprocessing), self.epoch_num, max_total_batch)

                start = time.time()
                self.keras_model.fit(keras_train_x, keras_train_y,
                              validation_data=(keras_valid_x, keras_valid_y),
                              epochs=self.epoch_num, batch_size=self.batch_size, callbacks=[checkpointer, losses], verbose=1, shuffle=True)
                print("{} training finishes in {:.2f} seconds.".format(b, time.time() - start))

                self.keras_model.load_weights("{}saved_models/{}/{}/keras_{}_{}_{}by{}_{}_weights.hdf5".format(self.root, self.network_type, device, b, self.dataset, self.resize_size[0], self.resize_size[1], self.preprocessing)) # Load the best model (not necessary the latest one)
                keras_predictions = [np.argmax(self.keras_model.predict(np.expand_dims(feature, axis=0))) for feature in keras_test_x]

                # report test accuracy
                keras_test_accuracy = 100 * np.sum(np.array(keras_predictions)==np.argmax(keras_test_y, axis=1))/len(keras_predictions)
                losses.f['.']['infer_acc']['accuracy'][0] = np.float32(keras_test_accuracy)
                losses.f.close()
                print('{} test accuracy: {:.1f}%'.format(b, keras_test_accuracy))

                json_string = self.keras_model.to_json()
                js = open("{}saved_models/{}/{}/keras_{}_{}_{}by{}_{}_config.json".format(self.root, self.network_type, device, b, self.dataset, self.resize_size[0], self.resize_size[1], self.preprocessing), "w")
                js.write(json_string)
                js.close()
