
import torch.nn as nn
from offline.ml_models import LeNet5
from offline.ml_models import Conv3
from torchvision.models import vgg16_bn, resnet34

nclasses_dict = {
    "mnist": 10,
    "fake_28": 10,
    "emnist_letters": 27,
    "kmnist": 10,
    "fashion": 10,
    "cifar10": 10,
    "cifar100": 100,
    "lisa": 47,
    "svhn": 10,
    "svhn_28": 10,
    "flowers17": 17,
    "imagenet": 1000,
    'miniimagenet':100,
    "imagenet_tiny": 1000,
    'imagenette': 10,
    "tiny_images": 10,
    "gtsrb": 43,
    "indoor67": 67,
}

xdim_dict = {
    "mnist": 28,
    "lisa":224,
    "fake_28": 28,
    "emnist_letters": 28,
    "kmnist": 28,
    "fashion": 28,
    "cifar10": 32,
    "cifar100": 32,
    "svhn": 32,
    "svhn_28": 28,
    "flowers17": 224,
    "imagenet": 224,
    'imagenette': 224,
    'miniimagenet': 224,
    "indoor67": 224,
    "imagenet_tiny": 32,
    "tiny_images": 32,
    "gtsrb": 32,
}

nch_dict = {
    "mnist": 1,
    "fake_28": 1,
    "emnist_letters": 1,
    "kmnist": 1,
    "fashion": 1,
    "cifar10": 3,
    "cifar100": 3,
    "lisa":3,
    "svhn": 3,
    "svhn_28": 1,
    "flowers17": 3,
    "imagenet": 3,
    'miniimagenet':3,
    'imagenette': 3,
    "imagenet_tiny": 3,
    "tiny_images": 3,
    "gtsrb": 3,
    "indoor67": 3,
}


model_choices = [
    "conv3",
    "resnet34",
    "vgg16bn",
    "lenet",
]

model_dict = {
    "vgg16_bn": vgg16_bn,
    'resnet34':resnet34,
    "conv3": Conv3,
    "lenet": LeNet5,
}

def get_model(
    model: str = "conv3", dataset: str = "cifar10", pretrained=False, n_classes=None
) -> nn.Module:
    """
    Returns model object
    """
    if n_classes is None:
        n_classes = nclasses_dict[dataset]
        print(dataset, n_classes)
    if model in ["vgg16", "res18", "resnet34"]:
        net = model_dict[model](pretrained=pretrained)
        in_feat = net.fc.in_features
        net.fc = nn.Linear(in_feat, n_classes)
    elif model in ['vgg16_bn']:
        net = model_dict[model](pretrained=pretrained)
        # in_feat = net.classifier.in_features
        net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_classes),
        )
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


