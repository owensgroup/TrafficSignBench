import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
import time, sys, DLHelper

print("**********************************")
print("Training on PyTorch")
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

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import torch.nn.init as torch_init
from torchvision import datasets, transforms
from torch.autograd import Variable
from timeit import default_timer
from pytorch_resnet import resnet

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class IDSIANet(torch.nn.Module):
    def __init__(self):
        super(IDSIANet, self).__init__()

        # Build model
        self.conv = torch.nn.Sequential()
        self.conv.add_module("torch_conv1", torch.nn.Conv2d(3, 100, kernel_size=(3, 3), stride=1))
        self.conv.add_module("torch_pool1", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("torch_relu1", torch.nn.ReLU())
        self.conv.add_module("torch_conv2", torch.nn.Conv2d(100, 150, kernel_size=(4, 4), stride=1))
        self.conv.add_module("torch_pool2", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("torch_relu2", torch.nn.ReLU())
        self.conv.add_module("torch_conv3", torch.nn.Conv2d(150, 250, kernel_size=(3, 3), stride=1))
        self.conv.add_module("torch_pool3", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("torch_relu3", torch.nn.ReLU())
        self.conv.add_module("torch_flatten", Flatten())

        
        self.csf = torch.nn.Sequential()
        self.csf.add_module("torch_fc1", torch.nn.Linear(250*4*4, 200))
        self.csf.add_module("torch_relu3", torch.nn.ReLU())
        self.csf.add_module("torch_fc2", torch.nn.Linear(200, class_num))
        
        # Initialize conv layers and fc layers
        torch_init.kaiming_normal(self.conv.state_dict()["torch_conv1.weight"])
        torch_init.constant(self.conv.state_dict()["torch_conv1.bias"], 0.0)
        torch_init.kaiming_normal(self.conv.state_dict()["torch_conv2.weight"])
        torch_init.constant(self.conv.state_dict()["torch_conv2.bias"], 0.0)
        torch_init.kaiming_normal(self.conv.state_dict()["torch_conv3.weight"])
        torch_init.constant(self.conv.state_dict()["torch_conv3.bias"], 0.0)
        torch_init.kaiming_normal(self.csf.state_dict()["torch_fc1.weight"])
        torch_init.constant(self.csf.state_dict()["torch_fc1.bias"], 0.0)
        torch_init.kaiming_normal(self.csf.state_dict()["torch_fc2.weight"])
        torch_init.constant(self.csf.state_dict()["torch_fc2.bias"], 0.0)

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 250*4*4)
        return self.csf.forward(x)

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Build model
        self.conv = torch.nn.Sequential()
        self.conv.add_module("torch_conv1", torch.nn.Conv2d(3, 64, kernel_size=(5, 5), stride=2, padding=2))
        self.conv.add_module("torch_pool1", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("torch_relu1", torch.nn.ReLU())
        self.conv.add_module("torch_conv2", torch.nn.Conv2d(64, 256, kernel_size=(3, 3), stride=1, padding=1))
        self.conv.add_module("torch_pool2", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("torch_relu2", torch.nn.ReLU())
        self.conv.add_module("torch_flatten", Flatten())

        
        self.csf = torch.nn.Sequential()
        self.csf.add_module("torch_fc1", torch.nn.Linear(256*6*6, 2048))
        self.csf.add_module("torch_relu3", torch.nn.ReLU())
        self.csf.add_module("torch_dropout1", torch.nn.Dropout(0.5))
        self.csf.add_module("torch_fc2", torch.nn.Linear(2048, class_num))
        
        # Initialize conv layers and fc layers
        torch_init.normal(self.conv.state_dict()["torch_conv1.weight"], mean=0, std=0.01)
        torch_init.constant(self.conv.state_dict()["torch_conv1.bias"], 0.0)
        torch_init.normal(self.conv.state_dict()["torch_conv2.weight"], mean=0, std=0.01)
        torch_init.constant(self.conv.state_dict()["torch_conv2.bias"], 0.0)
        torch_init.normal(self.csf.state_dict()["torch_fc1.weight"], mean=0, std=0.01)
        torch_init.constant(self.csf.state_dict()["torch_fc1.bias"], 0.0)
        torch_init.normal(self.csf.state_dict()["torch_fc2.weight"], mean=0, std=0.01)
        torch_init.constant(self.csf.state_dict()["torch_fc2.bias"], 0.0)

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 256*6*6)
        return self.csf.forward(x)

def constructCNN(cnn_type='self', gpu=True):
    torch_model_cpu, torch_model_gpu = None, None
    if cnn_type == "idsia":
        torch_model_cpu = IDSIANet()
        if gpu:
            torch_model_gpu = IDSIANet().cuda()
    elif cnn_type == "self":
        torch_model_cpu = ConvNet()
        if gpu:
            torch_model_gpu = ConvNet().cuda()
    elif cnn_type == 'resnet-56':
        torch_model_cpu = resnet(depth=56, num_classes=class_num)
        if gpu:
            torch_model_gpu = resnet(depth=56, num_classes=class_num).cuda()
    elif cnn_type == 'resnet-32':
        torch_model_cpu = resnet(depth=32, num_classes=class_num)
        if gpu:
            torch_model_gpu = resnet(depth=32, num_classes=class_num).cuda()
    elif cnn_type == 'resnet-20':
        torch_model_cpu = resnet(depth=20, num_classes=class_num)
        if gpu:
            torch_model_gpu = resnet(depth=20, num_classes=class_num).cuda()

    return torch_model_cpu, torch_model_gpu

torch_train_x = torch.stack([torch.Tensor(i.swapaxes(0,2).astype("float32")/255) for i in x_train])
torch_train_y = torch.LongTensor(y_train)
torch_valid_x = torch.stack([torch.Tensor(i.swapaxes(0,2).astype("float32")/255) for i in x_valid])
torch_valid_y = torch.LongTensor(y_valid)
torch_test_x = torch.stack([torch.Tensor(i.swapaxes(0,2).astype("float32")/255) for i in testImages])
torch_test_y = torch.LongTensor(testLabels)

torch_tensor_train_set = utils.TensorDataset(torch_train_x, torch_train_y)
torch_train_set = utils.DataLoader(torch_tensor_train_set, batch_size=batch_size, shuffle=True)
torch_tensor_valid_set = utils.TensorDataset(torch_valid_x, torch_valid_y)
torch_valid_set = utils.DataLoader(torch_tensor_valid_set, batch_size=batch_size, shuffle=True)
torch_tensor_test_set = utils.TensorDataset(torch_test_x, torch_test_y)
torch_test_set = utils.DataLoader(torch_tensor_test_set, batch_size=batch_size, shuffle=True)

torch_model_cpu, torch_model_gpu = constructCNN(network_type, gpu=("gpu" in backends))
max_total_batch = (len(x_train) // batch_size + 1) * epoch_num

model_parameters = filter(lambda p: p.requires_grad, torch_model_cpu.parameters())
prm = sum([np.prod(p.size()) for p in model_parameters])

def train(torch_model, optimizer, train_set, f, batch_count, gpu=False, epoch=None):
    if gpu:
        torch_model.cuda()
    torch_model.train() # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_set):
        batch_count += 1
        start = default_timer()
        if gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = torch_model(data)
        # output = torch_model.forward(data)
        cost = torch.nn.CrossEntropyLoss(size_average=True)
        train_loss = cost(output, target)
        train_loss.backward()
        optimizer.step()

        # Get the accuracy of this batch
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        acc = (100. * correct / len(data)) if (len(data) == batch_size) else f['.']['accuracy']['train'][batch_count-2]

        # Save batch time
        f['.']['time']['train_batch'][batch_count-1] = default_timer() - start

        # Save training loss and accuracy
        f['.']['cost']['train'][batch_count-1] = np.float32(train_loss.data[0])
        f['.']['accuracy']['train'][batch_count-1] = np.float32(acc)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},\tAccuracy: {}/{} ({:.0f}%)'.format(
                epoch, batch_idx * len(data), len(train_set.dataset),\
                100. * batch_idx / len(train_set), train_loss.data[0],\
                correct, len(data), acc))

    # Save batch marker
    f['.']['time_markers']['minibatch'][epoch] = np.float32(batch_count)

    return batch_count

def valid(torch_model, optimizer, valid_set, f, gpu = False, epoch = None):
    torch_model.eval() # Set the model to testing mode
    valid_loss = 0
    correct = 0
    for data, target in valid_set:
        if gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=False), Variable(target)
        output = torch_model(data)
        # output = torch_model.forward(data)
        cost = torch.nn.CrossEntropyLoss(size_average=False)
        valid_loss += cost(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    valid_loss /= len(valid_set.dataset)

    epoch_str = ""
    if epoch is not None:
        # Save validation loss and accuracy
        f['.']['cost']['loss'][epoch] = np.float32(valid_loss)
        f['.']['accuracy']['valid'][epoch] = np.float32(100. * correct / len(valid_set.dataset))
        epoch_str = "\nValid Epoch: {} ".format(epoch)
    else:
        # Save inference accuracy
        f['.']['infer_acc']['accuracy'][0] = np.float32(100. * correct / len(valid_set.dataset))
        epoch_str = "Test set: "
    print(epoch_str + 'Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_set.dataset),
        100. * correct / len(valid_set.dataset)))

# CPU & GPU
for b in backends:
    print("Run on {}".format(b))
    use_gpu = (b == 'gpu')
    batch_count = 0
    torch_model = torch_model_gpu if use_gpu else torch_model_cpu
    optimizer = optim.SGD(torch_model.parameters(), lr=0.01, momentum=0.9)

    filename = "{}saved_data/{}/{}/callback_data_pytorch_{}_{}by{}_{}.h5".format(root, network_type, b, dataset, resize_size[0], resize_size[0], process)
    f = DLHelper.init_h5py(filename, epoch_num, max_total_batch)
    try:
        f['.']['time']['train']['start_time'][0] = time.time()
        for epoch in range(epoch_num):

            # Start training and save start and end time
            batch_count = train(torch_model, optimizer, torch_train_set, f, batch_count, use_gpu, epoch)

            # Validation per epoch
            valid(torch_model, optimizer, torch_valid_set, f, use_gpu, epoch)
            
        f['.']['time']['train']['end_time'][0] = time.time()
        print(f['.']['time']['train']['end_time'][0] - f['.']['time']['train']['start_time'][0])

        # Save total batch count
        f['.']['config'].attrs["total_minibatches"] = batch_count
        f['.']['time_markers'].attrs['minibatches_complete'] = batch_count

        # Final test
        valid(torch_model, optimizer, torch_test_set, f, use_gpu)

        torch.save(torch_model, "{}saved_models/{}/{}/pytorch_{}_{}by{}_{}.pth".format(root, network_type, b, dataset, resize_size[0], resize_size[0], process))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e
    finally:
        print("Close file descriptor")
        f.close()
