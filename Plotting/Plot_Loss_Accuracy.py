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
parser.add_argument("--network_type", help="Type of neural network model", 
    type=str, default="idsia")
parser.add_argument("--resize_side", help="Size of the resized image", 
    type=int, default=48)
parser.add_argument("--dataset", help="Dataset", 
    type=str, default="GT")
parser.add_argument("-d", "--devices", action="append", help="Device (CPU/GPU)", 
    type=str, required=True)
parser.add_argument("--preprocessing", help="Preprocessing type", 
        type=str)
args = parser.parse_args()

root = args.root
dataset = args.dataset
model = args.network_type
size_xy = args.resize_side
devices = args.devices
data_path = root + "/saved_data/" + model

_ = DLHelper.create_dir(root, ["pics"], model, devices)

gpu_backends = ["cntk", "neon_gpu", "mxnet", "pytorch", "keras_tensorflow"]
cpu_backends = ["cntk", "neon_mkl", "mxnet", "pytorch", "keras_tensorflow"]

colors = {'neon_gpu': 'royalblue', 'neon_mkl': 'g', 'keras_tensorflow': 'r',\
    'keras_theano': 'c', 'cntk': 'm', 'mxnet': 'orange', 'pytorch': 'saddlebrown'}

markers = {'neon_gpu': 'o', 'neon_mkl': 'x', 'keras_tensorflow': 'p',\
    'keras_theano': 'v', 'cntk': '+', 'mxnet': 's', 'pytorch': '*'}

ylim_high = {'idsia': 4, 'self': 4, 'resnet-32': 3, 'resnet-20': 3}
preprocessings = {'none': 0, '1-sigma': 1, '2-sigma': 2, 'clahe': 3}
pre = preprocessings[args.preprocessing]

pics_num = 5

print("Model: {}".format(model))
for device in devices:
    pics_path = root + "/pics/" + model + "/" + device
    figs = [None] * pics_num
    axes = [None] * pics_num
    table = []

    subplots_num = (2, 3) if device == 'gpu' else (3, 3)
    figs[0] = plt.figure()
    axes[0] = figs[0].add_subplot(1,1,1)
    figs[0].suptitle("Training loss versus time ({})".format(device), y=0.94)
    figs[1], axes[1] = plt.subplots(subplots_num[0], subplots_num[1])
    figs[1].suptitle("Training and validation loss versus epoch ({})".format(device), y=0.94)

    figs[2] = plt.figure()
    axes[2] = figs[2].add_subplot(1,1,1)
    figs[2].suptitle("Training accuracy versus time ({})".format(device), y=0.94)
    figs[3] = plt.figure()
    axes[3] = figs[3].add_subplot(1,1,1)
    figs[3].suptitle("Validation accuracy versus time and Inference accuracy ({})".format(device), y=0.94)
    figs[4], axes[4] = plt.subplots(subplots_num[0], subplots_num[1])
    figs[4].suptitle("Training and validation accuracy versus epoch ({})".format(device), y=0.94)

    backends = gpu_backends if device == "gpu" else cpu_backends
    infer_acc = []
    fastest = None
    # print(device)
    for i in range(len(backends)):
        b = backends[i]
        filename = data_path+"/{}/callback_data_{}_{}_{}by{}_{}.h5".format(device, b, dataset, size_xy, size_xy, pre)
        print(filename)
        if not os.path.exists(filename):
            print("Data file for {} doesn't exist! Skip...".format(b))
            continue
        print(filename)

        train_cost_batch = pd.DataFrame()
        train_cost_epoch = pd.DataFrame()
        valid_cost_epoch = pd.DataFrame()
        train_acc_batch = pd.DataFrame()
        train_acc_epoch = pd.DataFrame()
        valid_acc_epoch = pd.DataFrame()
        train_epoch_mark = dict()
        
        # print(data_path)
        f = h5py.File(filename, "r")
        actual_length = f['.']['config'].attrs['total_minibatches']
        
        train_cost_batch['{}_loss'.format(b)] = pd.Series(f['.']['cost']['train'][()]).iloc[0:actual_length] # Training loss per batch
        train_cost_batch['{}_t'.format(b)] = pd.Series(f['.']['time']['train_batch'][()]).cumsum().iloc[0:actual_length] # Training time cumsum per batch
        
        valid_cost_epoch['{}_loss'.format(b)] = pd.Series(f['.']['cost']['loss'][()])
        valid_cost_epoch['{}_t'.format(b)] = pd.Series(f['.']['time']['loss'][()])

        train_acc_batch['{}_acc'.format(b)] = pd.Series(f['.']['accuracy']['train'][()]).iloc[0:actual_length] # Training accuracy per batch
        train_acc_batch['{}_t'.format(b)] = pd.Series(f['.']['time']['train_batch'][()]).cumsum().iloc[0:actual_length] # Training time cumsum per batch
        
        valid_acc_epoch['{}_acc'.format(b)] = pd.Series(f['.']['accuracy']['valid'][()])
        valid_acc_epoch['{}_t'.format(b)] = pd.Series(f['.']['time']['loss'][()])

        acc = f['.']['infer_acc']['accuracy'][0]
        infer_acc.append(acc)
        print((b, acc))

        tmp = (f['.']['time_markers']['minibatch'][()]-1).astype(int).tolist()
        tmp.pop()
        tmp = [0] + tmp
        train_epoch_mark['{}_mark'.format(b)] = tmp

        train_cost_epoch['{}_loss'.format(b)] = pd.Series([train_cost_batch['{}_loss'.format(b)][train_epoch_mark['{}_mark'.format(b)][j]:train_epoch_mark['{}_mark'.format(b)][j+1]].mean()\
            if j != (len(train_epoch_mark['{}_mark'.format(b)])-1)\
            else train_cost_batch['{}_loss'.format(b)][train_epoch_mark['{}_mark'.format(b)][j-1]:train_epoch_mark['{}_mark'.format(b)][j]].mean()\
            for j in range(0, len(train_epoch_mark['{}_mark'.format(b)]))
        ])
        train_cost_epoch['{}_t'.format(b)] = pd.Series(train_cost_batch['{}_t'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]].tolist())

        train_acc_epoch['{}_acc'.format(b)] = pd.Series([train_acc_batch['{}_acc'.format(b)][train_epoch_mark['{}_mark'.format(b)][j]:train_epoch_mark['{}_mark'.format(b)][j+1]].mean()\
            if j != (len(train_epoch_mark['{}_mark'.format(b)])-1)\
            else train_acc_batch['{}_acc'.format(b)][train_epoch_mark['{}_mark'.format(b)][j-1]:train_epoch_mark['{}_mark'.format(b)][j]].mean()\
            for j in range(0, len(train_epoch_mark['{}_mark'.format(b)]))
        ])
        train_acc_epoch['{}_t'.format(b)] = pd.Series(train_acc_batch['{}_t'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]].tolist())

        # Get the fastest framework
        if fastest is None:
            fastest = (b, train_acc_epoch['{}_t'.format(b)].iloc[-1])
        else:
            if (train_acc_epoch['{}_t'.format(b)].iloc[-1] < fastest[1]):
                fastest = (b, train_acc_epoch['{}_t'.format(b)].iloc[-1])

        # Avg training cost per epoch
        axes[0].plot(train_cost_epoch['{}_t'.format(b)], train_cost_epoch['{}_loss'.format(b)],\
            marker=markers[b], color=colors[b])

        # Avg training cost vs valid cost per epoch
        axes[1][int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
            train_cost_epoch['{}_loss'.format(b)], label='{}_t'.format(b), marker=markers["neon_gpu"])
        # axes[1][int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
        #     valid_cost_epoch['{}_loss'.format(b)], label='{}_v'.format(b), marker=markers["neon_mkl"])
        axes[1][int(i/3)][i%3].legend(loc=4)
        axes[1][int(i/3)][i%3].yaxis.grid(linestyle='dashdot')
        axes[1][int(i/3)][i%3].set_xlabel('Epoch')
        axes[1][int(i/3)][i%3].set_ylabel('Loss')
        
        # Avg training acc per epoch
        axes[2].plot(train_acc_epoch['{}_t'.format(b)], -1*train_acc_epoch['{}_acc'.format(b)]+101,\
            marker=markers[b], color=colors[b])
        
        # Valid acc per epoch
        axes[3].plot(train_acc_epoch['{}_t'.format(b)], -1*valid_acc_epoch['{}_acc'.format(b)]+101,\
            marker=markers[b], color=colors[b])
        
        # Avg training acc vs valid acc per epoch
        axes[4][int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
            train_acc_epoch['{}_acc'.format(b)], label='{}_t'.format(b), marker=markers["neon_gpu"])
        # axes[4][int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
        #     valid_acc_epoch['{}_acc'.format(b)], label='{}_v'.format(b), marker=markers["neon_mkl"])
        axes[4][int(i/3)][i%3].legend(loc=4)
        axes[4][int(i/3)][i%3].yaxis.grid(linestyle='dashdot')
        axes[4][int(i/3)][i%3].set_xlabel('Epoch')
        axes[4][int(i/3)][i%3].set_ylabel('Accuracy (%)')
        
        f.close()

    axes[0].legend(loc=4)
    axes[0].axvline(fastest[1], linestyle='dashed', color='#777777')
    axes[0].text(fastest[1]*1.01, 1, "First Finished Training: " + fastest[0], size=12)
    axes[0].set_ylim((2 * 1e-5, ylim_high[model] + 2))
    axes[0].set_yscale('log')
    axes[0].yaxis.grid(linestyle='dashdot')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Loss')

    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles, [label[0:label.rfind('_')] for label in labels], loc=4)
    axes[2].axvline(fastest[1], linestyle='dashed', color='#777777')
    axes[2].text(fastest[1]*1.01, 12, "First Finished Training: " + fastest[0], size=12)
    axes[2].set_ylim((0.9,101))
    axes[2].set_yscale('log')
    axes[2].set_yticks([101-num for num in (list(range(0,11,1)) + list(range(11,91,10)) + list(range(91,101,1)))])
    axes[2].set_yticklabels(([''] * 11 + list(range(20,100,10)) + list(range(91,101,1))))
    axes[2].invert_yaxis()
    axes[2].yaxis.grid(linestyle='dashdot')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Training Accuracy (%)')

    handles, labels = axes[3].get_legend_handles_labels()
    axes[3].legend(handles, [label[0:label.rfind('_')] for label in labels], loc=4)
    axes[3].axvline(fastest[1], linestyle='dashed', color='#777777')
    axes[3].text(fastest[1]*1.01, 12, "First Finished Training: " + fastest[0], size=12)
    axes[3].set_ylim((0.9,101))
    axes[3].set_yscale('log')
    axes[3].set_yticks([101-num for num in (list(range(0,11,1)) + list(range(11,91,10)) + list(range(91,101,1)))])
    axes[3].set_yticklabels(([''] * 11 + list(range(20,100,10)) + list(range(91,101,1))))
    axes[3].invert_yaxis()
    axes[3].yaxis.grid(linestyle='dashdot')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Validation Accuracy (%)')

    figs[0].savefig(pics_path+"/train_loss_versus_time_{}by{}_{}.png".format(size_xy, size_xy, pre), dpi=figs[0].dpi)
    figs[1].savefig(pics_path+"/train_and_valid_loss_versus_epoch_{}by{}_{}.png".format(size_xy, size_xy, pre), dpi=figs[1].dpi)
    figs[2].savefig(pics_path+"/train_acc_versus_time_{}by{}_{}.png".format(size_xy, size_xy, pre), dpi=figs[2].dpi)
    figs[3].savefig(pics_path+"/valid_acc_versus_time_{}by{}_{}.png".format(size_xy, size_xy, pre), dpi=figs[3].dpi)
    figs[4].savefig(pics_path+"/train_and_valid_acc_versus_epoch_{}by{}_{}.png".format(size_xy, size_xy, pre), dpi=figs[4].dpi)
    
    # plt.show()
