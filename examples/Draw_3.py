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
data_path = root + "/saved_data/"
device = sys.argv[2]
title = int(sys.argv[4])

size = 48
fontsize = 16
models = ["IDSIA", "ResNet-20", "ResNet-32"]

for model in models:
	_ = DLHelper.create_dir(root, ["pics"], model, [device])

gpu_backends = ["cntk", "neon_gpu", "mxnet", "pytorch", "keras_tensorflow"]
cpu_backends = ["cntk", "neon_mkl", "mxnet", "pytorch", "keras_tensorflow"]

colors = {'neon_gpu': 'royalblue', 'neon_mkl': 'royalblue', 'keras_tensorflow': 'r',\
    'keras_theano': 'c', 'cntk': 'm', 'mxnet': 'orange', 'pytorch': 'saddlebrown'}

markers = {'neon_gpu': 'o', 'neon_mkl': 'x', 'keras_tensorflow': 'p',\
    'keras_theano': 'v', 'cntk': '+', 'mxnet': 's', 'pytorch': '*'}

correct_name = {'neon_gpu': 'Neon', 'neon_mkl': 'Neon',\
	'keras_tensorflow': 'Tensorflow', 'keras_theano': 'Theano', \
	'cntk': 'CNTK', 'mxnet': 'MXNet', 'pytorch': 'PyTorch'}

figs = [None] * 2
axes = [None] * 2

figs[0] = plt.figure()
axes[0] = figs[0].add_subplot(1,1,1)
if title == 1:
	figs[0].suptitle("Training time versus different models with input size of 48x48 ({})".format(device.upper()), y=0.94)

figs[1] = plt.figure()
axes[1] = figs[1].add_subplot(1,1,1)
if title == 1:
	figs[1].suptitle("Training time versus different models with input size of 48x48 ({})".format(device.upper()), y=0.94)

backends = gpu_backends if device == "gpu" else cpu_backends

for idy, b in enumerate(backends):
	ts = []
	for model in models:
		f = h5py.File(data_path+"/{}/{}/callback_data_{}_{}_{}by{}_3.h5".format(model, device, b, dataset, size, size), "r")
		actual_length = f['.']['config'].attrs['total_minibatches']

		ts.append(f['.']['time']['train_batch'][()].mean())
		f.close()
	axes[0].plot(range(3), ts, marker=markers[b], color=colors[b], label=correct_name[b])
	axes[1].bar([-0.2 + x + idy*0.1 for x in range(len(models))], ts, 0.1, color=colors[b], label=correct_name[b])
	
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, loc=4, fontsize=fontsize)
axes[0].yaxis.grid(linestyle='dashdot')
axes[0].set_xticks(range(3))
axes[0].set_xticklabels(models, fontsize=fontsize)
for tick in axes[0].yaxis.get_major_ticks():
	tick.label.set_fontsize(fontsize)
axes[0].set_xlabel('Models', fontsize=fontsize)
axes[0].set_ylabel('Training time per batch (s)', fontsize=fontsize)

handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles, labels, loc=2, fontsize=fontsize)
axes[1].yaxis.grid(linestyle='dashdot')
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(models, fontsize=fontsize)
for tick in axes[1].yaxis.get_major_ticks():
	tick.label.set_fontsize(fontsize)
axes[1].set_xlabel('Models', fontsize=fontsize)
axes[1].set_ylabel('Training time per batch (s)', fontsize=fontsize)

plt.tight_layout()

pics_path = root + "/pics/"
figs[0].savefig(pics_path+"/batch_training_time_48by48_plot_all_models_{}.png".format(device), dpi=figs[0].dpi)
figs[1].savefig(pics_path+"/batch_training_time_48by48_bar_all_models_{}.png".format(device), dpi=figs[1].dpi)

figs[1].savefig("/Users/moderato/Downloads/iv_paper/batch_training_time_48by48_bar_all_models_{}.png".format(device), dpi=figs[1].dpi)

plt.show()