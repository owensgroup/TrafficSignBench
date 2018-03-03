# Code ported from Neon official Model Zoo

from neon.initializers import Kaiming, IdentityInit
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, Activation
from neon.layers import MergeSum, SkipNode
from neon.transforms import Rectlin, Softmax

def conv_params(fsize, nfm, stride=1, relu=True):
    return dict(fshape=(fsize, fsize, nfm), strides=stride, padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                init=Kaiming(local=True),
                batch_norm=True)


def id_params(nfm):
    return dict(fshape=(1, 1, nfm), strides=2, padding=0, activation=None, init=IdentityInit())


def module_factory(nfm, stride=1):
    mainpath = [Conv(**conv_params(3, nfm, stride=stride)),
                Conv(**conv_params(3, nfm, relu=False))]
    sidepath = [SkipNode() if stride == 1 else Conv(**id_params(nfm))]
    module = [MergeSum([mainpath, sidepath]),
              Activation(Rectlin())]
    return module


def resnet(depth, num_classes, s):
    # Structure of the deep residual part of the network:
    # args.depth modules of 2 convolutional layers each at feature map depths of 16, 32, 64
    nfms = [2**(stage + 4) for stage in sorted(list(range(3)) * depth)]
    strides = [1] + [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

    # Now construct the network
    layers = [Conv(**conv_params(3, 16))]
    for nfm, stride in zip(nfms, strides):
        layers.append(module_factory(nfm, stride))
    layers.append(Pooling(s, op='avg'))
    layers.append(Affine(nout=num_classes, init=Kaiming(local=False), batch_norm=True, activation=Softmax()))

    return layers
