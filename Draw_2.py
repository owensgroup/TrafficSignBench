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
model = sys.argv[2]
if model not in ['resnet-20', 'resnet-32']:
	raise Exception();
data_path = root + "/saved_data/" + model
device = sys.argv[3]
fontsize = int(sys.argv[4])
title = int(sys.argv[5])

sizes = [32, 48, 64]

_ = DLHelper.create_dir(root, ["pics"], model, [device])

gpu_backends = ["cntk", "neon_gpu", "mxnet", "pytorch", "keras_tensorflow"]
cpu_backends = ["cntk", "neon_mkl", "mxnet", "pytorch", "keras_tensorflow"]

colors = {'neon_gpu': 'royalblue', 'neon_mkl': 'royalblue', 'keras_tensorflow': 'r',\
    'keras_theano': 'c', 'cntk': 'm', 'mxnet': 'orange', 'pytorch': 'saddlebrown'}

markers = {'neon_gpu': 'o', 'neon_mkl': 'x', 'keras_tensorflow': 'p',\
    'keras_theano': 'v', 'cntk': '+', 'mxnet': 's', 'pytorch': '*'}

correct_name = {'neon_gpu': 'Neon', 'neon_mkl': 'Neon', \
	'keras_tensorflow': 'Tensorflow', 'keras_theano': 'Theano', \
	'cntk': 'CNTK', 'mxnet': 'MXNet', 'pytorch': 'PyTorch'}

model_name = {"resnet-32": "ResNet-32", "resnet-20": "ResNet-20", "idsia": "IDSIA"}

figs = [None] * 2
axes = [None] * 2

figs[0] = plt.figure()
axes[0] = figs[0].add_subplot(1,1,1)
if title == 1:
	figs[0].suptitle("Training time versus input sizes on {} ({})".format(model_name[model], device.upper()), y=0.94)

figs[1] = plt.figure()
axes[1] = figs[1].add_subplot(1,1,1)
if title == 1:
	figs[1].suptitle("Training time versus input sizes on {} ({})".format(model_name[model], device.upper()), y=0.94)

backends = gpu_backends if device == "gpu" else cpu_backends

for idy, b in enumerate(backends):
	ts = []
	for size in sizes:
		filename = data_path+"/{}/callback_data_{}_{}_{}by{}_3.h5".format(device, b, dataset, size, size)
		f = h5py.File(filename, "r")
		actual_length = f['.']['config'].attrs['total_minibatches']

		ts.append(f['.']['time']['train_batch'][()].mean())
		f.close()
	axes[0].plot(range(3), ts, marker=markers[b], color=colors[b], label=correct_name[b])
	axes[1].bar([-0.2 + x + idy*0.1 for x in range(len(sizes))], ts, 0.1, color=colors[b], label=correct_name[b])
	

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, loc=4, fontsize=fontsize)
axes[0].yaxis.grid(linestyle='dashdot')
axes[0].set_xticks(range(3))
axes[0].set_xticklabels(["{}x{}".format(32 + 16*x, 32 + 16*x) for x in range(3)], fontsize=fontsize)
for tick in axes[0].yaxis.get_major_ticks():
	tick.label.set_fontsize(fontsize) 
axes[0].set_xlabel('Input image size', fontsize=fontsize)
axes[0].set_ylabel('Training time per batch (s)', fontsize=fontsize)

handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles, labels, loc=2, fontsize=fontsize)
axes[1].yaxis.grid(linestyle='dashdot')
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(["{}x{}".format(32 + 16*x, 32 + 16*x) for x in range(3)], fontsize=fontsize)
for tick in axes[1].yaxis.get_major_ticks():
	tick.label.set_fontsize(fontsize) 
axes[1].set_xlabel('Input image size', fontsize=fontsize)
axes[1].set_ylabel('Training time per batch (s)', fontsize=fontsize)

plt.tight_layout()

pics_path = root + "/pics/" + model + "/" + device
figs[0].savefig(pics_path+"/batch_training_time_all_input_sizes_plot_{}_{}.png".format(model, device), dpi=figs[0].dpi)
figs[1].savefig(pics_path+"/batch_training_time_all_input_sizes_bar_{}_{}.png".format(model, device), dpi=figs[1].dpi)

figs[1].savefig("/Users/moderato/Downloads/iv_paper/batch_training_time_all_input_sizes_bar_{}_{}.png".format(model, device), dpi=figs[1].dpi)

plt.show()
