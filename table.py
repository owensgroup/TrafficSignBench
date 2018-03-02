import matplotlib
import matplotlib.pyplot as plt
import h5py, sys, os
import DLHelper
# %matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10,8)

if sys.platform == "darwin":
    root = "/Users/moderato/Downloads"
else:
    root = "/home/zhongyilin/Desktop"
dataset = sys.argv[1]
if dataset == "GT":
    root += "/GTSRB/try"
size_xy = int(sys.argv[2])
if size_xy not in [32, 48, 64]:
	raise Exception();
models = ['resnet-20', 'resnet-32']
if size_xy == 48:
	models = ['idsia'] + models
devices = sys.argv[3:]

gpu_backends = ["cntk", "neon_gpu", "mxnet", "pytorch", "keras_tensorflow"]
cpu_backends = ["cntk", "neon_mkl", "mxnet", "pytorch", "keras_tensorflow"]

correct_name = {'neon_gpu': 'Neon', 'neon_mkl': 'Neon', \
	'keras_tensorflow': 'Tensorflow', 'keras_theano': 'Theano', \
	'cntk': 'CNTK', 'mxnet': 'MXNet', 'pytorch': 'PyTorch'}

lines = [correct_name[b] for b in gpu_backends]

for device in devices:
	backends = gpu_backends if device == "gpu" else cpu_backends
	for model in models:
		data_path = root + "/saved_data/" + model
		for idx, b in enumerate(backends):
			f = h5py.File(data_path+"/{}/callback_data_{}_{}_{}by{}_3.h5".format(device, b, dataset, size_xy, size_xy), "r")

			t_train = f['.']['time']['train_batch'][()].sum()
			# t_total = f['.']['time']['train']['end_time'][0] - f['.']['time']['train']['start_time'][0]
			accuracy = f['.']['infer_acc']['accuracy'][0]

			# Text for latex table
			lines[idx] += "\t& {:.2f}\t& {:.2f}\%".format(t_train, accuracy)


for idx in range(len(lines)):
	lines[idx] += "\t\\\\ \\hline"
	print(lines[idx])

