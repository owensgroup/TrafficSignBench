import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import h5py, sys, os, argparse
sys.path.append("..")
import DLHelper
# %matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10,8)

parser = argparse.ArgumentParser()

parser.add_argument("--root", help="Root directory of the dataset", 
	type=str, required=True)
parser.add_argument("--dataset", help="Dataset", 
	type=str, default="GT")
parser.add_argument("-d", "--devices", action="append", help="Device (CPU/GPU)", 
	type=str, required=True)
parser.add_argument("--preprocessing", help="Preprocessing type", 
	type=str)
parser.add_argument("--title", help="Print title or not", 
	type=int, default=1)
args = parser.parse_args()

root = args.root
dataset = args.dataset
if dataset == "GT":
    root = root + "/GTSRB"
devices = args.devices
title = args.title
data_path = root + "/saved_data"

size = 48
fontsize = 16
models = ["IDSIA", "ResNet-20", "ResNet-32"]

for model in models:
	_ = DLHelper.create_dir(root, ["pics"], model, devices)

gpu_backends = ["cntk", "neon_gpu", "mxnet", "pytorch", "keras_tensorflow"]
cpu_backends = ["cntk", "neon_mkl", "mxnet", "pytorch", "keras_tensorflow"]

colors = {'neon_gpu': 'royalblue', 'neon_mkl': 'royalblue', 'keras_tensorflow': 'r',\
	'keras_theano': 'c', 'cntk': 'm', 'mxnet': 'orange', 'pytorch': 'saddlebrown'}

markers = {'neon_gpu': 'o', 'neon_mkl': 'x', 'keras_tensorflow': 'p',\
	'keras_theano': 'v', 'cntk': '+', 'mxnet': 's', 'pytorch': '*'}

correct_name = {'neon_gpu': 'Neon', 'neon_mkl': 'Neon',\
	'keras_tensorflow': 'Tensorflow', 'keras_theano': 'Theano', \
	'cntk': 'CNTK', 'mxnet': 'MXNet', 'pytorch': 'PyTorch'}

for device in devices:
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

	# Plot average batch training time
	for idy, b in enumerate(backends):
		ts = []
		for model in models:
			filename = data_path+"/{}/{}/callback_data_{}_{}_{}by{}_3.h5".format(model.lower(), device, b, dataset, size, size)
			# print(filename)
			if not os.path.exists(filename):
				print("Data file for {} doesn't exist! Skip...".format(b))
				continue
			f = h5py.File(filename, "r")
			actual_length = f['.']['config'].attrs['total_minibatches']

			ts.append(f['.']['time']['train_batch'][()].mean())
			f.close()
		axes[0].plot(range(3), ts, marker=markers[b], color=colors[b], label=correct_name[b])
		axes[1].bar([-0.2 + x + idy*0.1 for x in range(len(models))], ts, 0.1, color=colors[b], label=correct_name[b])
		
	# Add labels, grids, legends, etc
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

	# Save plots
	pics_path = root + "/pics/"
	figs[0].savefig(pics_path+"/batch_training_time_48by48_plot_all_models_{}.png".format(device), dpi=figs[0].dpi)
	figs[1].savefig(pics_path+"/batch_training_time_48by48_bar_all_models_{}.png".format(device), dpi=figs[1].dpi)

	# plt.show()