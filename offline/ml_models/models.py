"""This module defines functions related to DNN models"""

import torch.nn as nn
from .conv3 import Conv3
from .resnet import resnet20, resnet32
from .lenet import LeNet5
from .wresnet import WideResNet
# from ml_datasets import nclasses_dict, nch_dict, xdim_dict
from functools import partial
from torchvision.models import vgg16, resnet18, mobilenet_v2

model_choices = [
    "conv3",
    "res18",
    "vgg16",
    "res20",
    "res32",
    "wres16-4",
    "wres16-8",
    "wres22-1",
    "wres22-4",
    "wres22-8",
    "wres28-10",
    "lenet",
    "mobilenet",
]

model_dict = {
    "vgg16": vgg16,
    "res18": resnet18,
    "conv3": Conv3,
    "res20": resnet20,
    "res32": resnet32,
    "wres16-4": partial(WideResNet, 16, widen_factor=4),
    "wres16-8": partial(WideResNet, 16, widen_factor=8),
    "wres22-1": partial(WideResNet, 22, widen_factor=1),
    "wres22-4": partial(WideResNet, 22, widen_factor=4),
    "wres22-8": partial(WideResNet, 22, widen_factor=8),
    "wres28-10": partial(WideResNet, 28, widen_factor=10),
    "lenet": LeNet5,
    "mobilenet": mobilenet_v2,
}

torchvision_models = ["vgg16", "res18"]


def get_model(
    model: str = "conv3", dataset: str = "cifar10", pretrained=False, n_classes=None
) -> nn.Module:
    """
    Returns model object
    """
    if n_classes is None:
        n_classes = nclasses_dict[dataset]
    if model in ["vgg16", "res18"]:
        net = model_dict[model](pretrained=pretrained)
        in_feat = net.fc.in_features
        net.fc = nn.Linear(in_feat, n_classes)
    elif model in ["mobilenet"]:
        net = model_dict[model](pretrained=pretrained)
        in_feat = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_feat, n_classes)
    else:
        try:
            net = model_dict[model](xdim_dict[dataset], nch_dict[dataset], n_classes)
        except:
            raise ValueError(f"model: {model} for dataset: {dataset} is undefined")
    return net

