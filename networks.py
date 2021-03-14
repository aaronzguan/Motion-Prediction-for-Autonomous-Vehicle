import torch
import torch.nn as nn
import torchvision.models as torch_models


def Make_layer(block, num_filters, num_of_layer):
    layers = []
    for _ in range(num_of_layer):
        layers.append(block(num_filters))
    return nn.Sequential(*layers)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def resnet28(feature_dim, use_pool, use_dropout):
    class ResBlock(nn.Module):
        def __init__(self, channels):
            super(ResBlock, self).__init__()
            self.resblock = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.PReLU(channels)
            )
        def forward(self, x):
            return x + self.resblock(x)

    filters = [64, 128, 256, 512]
    units = [1, 2, 5, 3]
    net_list = []
    for i, (num_units, num_filters) in enumerate(zip(units, filters)):
        if i == 0:
            net_list += [nn.Conv2d(3, 64, 3),
                         nn.PReLU(64),
                         nn.Conv2d(64, 64, 3),
                         nn.PReLU(64),
                         nn.MaxPool2d(2)]
        elif i == 1:
            net_list += [nn.Conv2d(64, 128, 3),
                         nn.PReLU(128),
                         nn.MaxPool2d(2)]
        elif i == 2:
            net_list += [nn.Conv2d(128, 256, 3),
                         nn.PReLU(256),
                         nn.MaxPool2d(2)]
        elif i == 3:
            net_list += [nn.Conv2d(256, 512, 3),
                         nn.PReLU(512),
                         nn.MaxPool2d(2)]
        if num_units > 0:
            net_list += [Make_layer(ResBlock, num_filters=num_filters, num_of_layer=num_units)]
    if use_pool:
        net_list += [nn.AdaptiveAvgPool2d((1, 1))]
    net_list += [Flatten()]
    if use_dropout:
        net_list += [nn.Dropout()]
    if use_pool:
        net_list += [nn.Linear(512, feature_dim)]
    else:
        net_list += [nn.Linear(512*5*4, feature_dim)]
    return nn.Sequential(*net_list)


def spherenet(num_layers, feature_dim, use_pool, use_dropout, use_lbp):
    """SphereNets.
    We follow the paper, and the official caffe code:
        SphereFace: Deep Hypersphere Embedding for Face Recognition, CVPR, 2017.
        https://github.com/wy1iu/sphereface
    """
    class SphereResBlock(nn.Module):
        def __init__(self, channels):
            super(SphereResBlock, self).__init__()
            self.resblock = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.PReLU(channels)
            )

        def forward(self, x):
            return x + self.resblock(x)

    filters = [64, 128, 256, 512]
    if num_layers == 4:
        units = [0, 0, 0, 0]
    elif num_layers == 10:
        units = [0, 1, 2, 0]
    elif num_layers == 20:
        units = [1, 2, 4, 1]
    elif num_layers == 36:
        units = [2, 4, 8, 2]
    elif num_layers == 64:
        units = [3, 8, 16, 3]
    net_list = []
    for i, (num_units, num_filters) in enumerate(zip(units, filters)):
        if i == 0:
            if use_lbp:
                net_list += [nn.Conv2d(1, 64, 3, 2, 1), nn.PReLU(64)]
            else:
                net_list += [nn.Conv2d(3, 64, 3, 2, 1), nn.PReLU(64)]
        elif i == 1:
            net_list += [nn.Conv2d(64, 128, 3, 2, 1), nn.PReLU(128)]
        elif i == 2:
            net_list += [nn.Conv2d(128, 256, 3, 2, 1), nn.PReLU(256)]
        elif i == 3:
            net_list += [nn.Conv2d(256, 512, 3, 2, 1), nn.PReLU(512)]
        if num_units > 0:
            net_list += [Make_layer(SphereResBlock, num_filters=num_filters, num_of_layer=num_units)]
    if use_pool:
        net_list += [nn.AdaptiveAvgPool2d((1, 1))]
    net_list += [Flatten()]
    if use_dropout:
        net_list += [nn.Dropout()]
    if use_pool:
        net_list += [nn.Linear(512, feature_dim)]
    else:
        net_list += [nn.Linear(512*7*6, feature_dim)]
    return nn.Sequential(*net_list)


def densenet121(feature_dim, use_pool, use_dropout):
    """
    We use the torchvision model for convenient.
    """
    net_list = list(list(torch_models.densenet121(pretrained=False).children())[:-1][0])
    net_list[0] = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if use_pool:
        net_list += [nn.AdaptiveAvgPool2d((1, 1))]
    net_list += [Flatten()]
    if use_dropout:
        net_list += [nn.Dropout()]
    if use_pool:
        net_list += [nn.Linear(1024, feature_dim)]
    else:
        # todo: need to test
        net_list += [nn.Linear(1024*3*3, feature_dim)]
    return nn.Sequential(*net_list)


def densenet161(feature_dim, use_pool, use_dropout):
    """
    We use the torchvision model for convenient.
    """
    net_list = list(list(torch_models.densenet161(pretrained=False).children())[:-1][0])
    net_list[0] = torch.nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=False)
    if use_pool:
        net_list += [nn.AdaptiveAvgPool2d((1, 1))]
    net_list += [Flatten()]
    if use_dropout:
        net_list += [nn.Dropout()]
    if use_pool:
        net_list += [nn.Linear(2208, feature_dim)]
    else:
        # todo: need to test
        net_list += [nn.Linear(2208*3*3, feature_dim)]
    return nn.Sequential(*net_list)

