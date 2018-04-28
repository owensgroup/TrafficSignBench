import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
import time, sys, DLHelper, random

import tensorflow as tf
import ResNet.tensorflow_resnet as tensorflow_resnet
from timeit import default_timer

class TensorflowBench:
	class IDSIA:
		def __init__(self, class_num):
			self.class_num = class_num

		def __call__(self, inputs, training):
			conv1 = tf.layers.conv2d(
				inputs=inputs,
				filters=100,
				kernel_size=[3, 3],
				padding="same",
				activation=tf.nn.relu,
				kernel_initializer=tf.keras.initializers.he_normal(seed=None),
				name="tf_conv1")
			pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="tf_pool1")
			conv2 = tf.layers.conv2d(
				inputs=pool1,
				filters=150,
				kernel_size=[4, 4],
				padding="same",
				activation=tf.nn.relu,
				kernel_initializer=tf.keras.initializers.he_normal(seed=None),
				name="tf_conv2")
			pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="tf_pool2")
			conv3 = tf.layers.conv2d(
				inputs=pool2,
				filters=250,
				kernel_size=[3, 3],
				padding="same",
				activation=tf.nn.relu,
				kernel_initializer=tf.keras.initializers.he_normal(seed=None),
				name="tf_conv3")
			pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name="tf_pool3")
			flatten = tf.layers.flatten(inputs=pool3, name="tf_flatten")
			dense1 = tf.layers.dense(inputs=flatten, 
				units=200, 
				kernel_initializer=tf.keras.initializers.he_normal(seed=None),
				name="tf_dense1")
			dense2 = tf.layers.dense(inputs=dense1, 
				units=self.class_num, 
				kernel_initializer=tf.keras.initializers.he_normal(seed=None),
				name="tf_dense2")
			return dense2

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

		self.tensorflow_model = None

		_ = DLHelper.create_dir(root, ["saved_data", "saved_models"], self.network_type, self.devices)

		print("**********************************")
		print("Training on Tensorflow")
		print("**********************************")

	def constructCNN(self):
		if self.network_type == "idsia":
			self.tensorflow_model = self.IDSIA(self.class_num)
			
		elif self.network_type == "resnet-56":
			self.tensorflow_model = tensorflow_resnet.Model(
				resnet_size=56,
				bottleneck=False,
				num_classes=self.class_num,
				num_filters=16,
				kernel_size=3,
				conv_stride=1,
				first_pool_size=None,
				first_pool_stride=None,
				second_pool_size=self.resize_size[0]/4,
				second_pool_stride=1,
				block_sizes=[9] * 3,
				block_strides=[1, 2, 2],
				final_size=64,
				version=1)
		elif self.network_type == "resnet-32":
			self.tensorflow_model = tensorflow_resnet.Model(
				resnet_size=32,
				bottleneck=False,
				num_classes=self.class_num,
				num_filters=16,
				kernel_size=3,
				conv_stride=1,
				first_pool_size=None,
				first_pool_stride=None,
				second_pool_size=self.resize_size[0]/4,
				second_pool_stride=1,
				block_sizes=[5] * 3,
				block_strides=[1, 2, 2],
				final_size=64,
				version=1)
		elif self.network_type == "resnet-20":
			self.tensorflow_model = tensorflow_resnet.Model(
				resnet_size=20,
				bottleneck=False,
				num_classes=self.class_num,
				num_filters=16,
				kernel_size=3,
				conv_stride=1,
				first_pool_size=None,
				first_pool_stride=None,
				second_pool_size=self.resize_size[0]/4,
				second_pool_stride=1,
				block_sizes=[3] * 3,
				block_strides=[1, 2, 2],
				final_size=64,
				version=1)

	def benchmark(self):
		# Prepare training/validation/testing sets
		tensorflow_train_x = np.vstack([np.expand_dims(np.asarray(x), axis=0).astype('float32')/255 for x in self.x_train]).astype('float32')
		tensorflow_valid_x = np.vstack([np.expand_dims(np.asarray(x), axis=0).astype('float32')/255 for x in self.x_valid]).astype('float32')
		tensorflow_test_x = np.vstack([np.expand_dims(np.asarray(x), axis=0).astype('float32')/255 for x in self.testImages]).astype('float32')
		tensorflow_train_y = tf.one_hot(self.y_train, self.class_num)
		tensorflow_valid_y = tf.one_hot(self.y_valid, self.class_num)
		tensorflow_test_y = tf.one_hot(self.testLabels, self.class_num)

		# Some constants
		max_total_batch = (len(self.x_train) // self.batch_size + 1) * self.epoch_num
		train_batch_num = len(self.x_train) // self.batch_size + 1
		valid_batch_num = len(self.x_valid) // self.batch_size + 1
		test_batch_num = len(self.testImages) // self.batch_size + 1

		# Construct model instance
		self.constructCNN()

		# Create file to save data
		filename = "{}/saved_data/{}/{}/callback_data_tensorflow_{}_{}by{}_{}.h5".format(self.root, self.network_type, self.devices[0], self.dataset, self.resize_size[0], self.resize_size[1], self.preprocessing)
		f = DLHelper.init_h5py(filename, self.epoch_num, train_batch_num * self.epoch_num)

		x = tf.placeholder(tf.float32, shape=[None, self.resize_size[0], self.resize_size[1], 3])
		training = tf.placeholder(tf.bool)
		y = tf.placeholder(tf.float32, shape=[None, self.class_num])
		logits = tf.cast(self.tensorflow_model(x, training), tf.float32)
		tensorflow_cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
		loss = tf.reduce_mean(tensorflow_cost)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float32))
		tensorflow_optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss)


		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			tf.get_variable_scope().reuse_variables()
			try:
				train_batch_count = 0
				f['.']['time']['train']['start_time'][0] = time.time()

				for epoch in range(0, self.epoch_num):
					is_training = True

					shuffled_indices = list(range(0, len(self.x_train)))
					random.shuffle(shuffled_indices)

					for i in range(0, train_batch_count):
						# Get data
						indices = shuffled_indices[i*self.batch_size:(i+1)*self.batch_size]
						features = tensorflow_train_x[indices, :, :, :]
						labels = tensorflow_train_y.eval()[indices, :]

						# Run
						start = default_timer()
						_, training_loss, training_acc = sess.run([tensorflow_optimizer, loss, accuracy], feed_dict={x: features, training: is_training, y: labels})
						train_batch_time = default_timer() - start

						# Save data
						f['.']['time']['train_batch'][train_batch_count-1] = train_batch_time
						train_batch_count += 1

						f['.']['cost']['train'][train_batch_count-1] = np.float32(training_loss)
						f['.']['accuracy']['train'][train_batch_count-1] = np.float32(training_acc * 100.0)

						if i % 30 == 0: # Print per 30 batches
							print("Epoch: {0}, Minibatch: {1}, Loss: {2:.4f}, Accuracy: {3:.2f}%".format(epoch, i, training_loss, training_acc * 100.0))

					# Save batch marker
					f['.']['time_markers']['minibatch'][epoch] = np.float32(train_batch_count)



					is_training = False
					shuffled_indices = list(range(0, len(self.x_valid)))
					random.shuffle(shuffled_indices)

					validation_loss = 0
					validation_acc = 0
					for i in range(0, valid_batch_num):
						indices = shuffled_indices[i*self.batch_size:(i+1)*self.batch_size]
						features = tensorflow_valid_x[indices, :, :, :]
						labels = tensorflow_valid_y.eval()[indices, :]

						# Get the loss and acc for this validation batch and accumulate
						l, a = sess.run([loss, accuracy], feed_dict={x: features, training: is_training, y: labels})
						validation_loss += l * labels.shape[0]
						validation_acc += a * labels.shape[0]

					validation_loss /= len(self.x_valid)
					validation_acc /= len(self.x_valid)

					f['.']['cost']['loss'][epoch] = np.float32(validation_loss)
					f['.']['accuracy']['valid'][epoch] = np.float32(validation_acc * 100.0)
					print("[Validation]")
					print("Epoch: {0}, Loss: {1:.4f}, Accuracy: {2:.2f}%".format(epoch, validation_loss, validation_acc * 100.0))

				# Save related params
				f['.']['time']['train']['end_time'][0] = time.time() # Save training time
				f['.']['config'].attrs["total_minibatches"] = train_batch_count
				f['.']['time_markers'].attrs['minibatches_complete'] = train_batch_count




				shuffled_indices = list(range(0, len(self.testImages)))
				random.shuffle(shuffled_indices)

				test_acc = 0
				for i in range(0, test_batch_num):
					indices = shuffled_indices[i*self.batch_size:(i+1)*self.batch_size]
					features = tensorflow_test_x[indices, :, :, :]
					labels = tensorflow_test_y.eval()[indices, :]

					# Get the acc for this testing batch and accumulate
					a = sess.run([accuracy], feed_dict={x: features, training: is_training, y: labels})
					test_acc += a[0] * labels.shape[0]

				test_acc /= len(self.testImages)

				# Save the testing accuracy
				f['.']['infer_acc']['accuracy'][0] = np.float32(test_acc * 100.0)
				print("Accuracy score is %f" % test_acc)


			except KeyboardInterrupt:
				pass
			except Exception as e:
				raise e
			finally:
				print("Close file descriptor")
				f.close()

