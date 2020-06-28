import torch.nn as nn
from torchvision import models


def get_model(name, pretrained, num_channels, num_classes):
    """
    Method that returns a torchvision model given a model
    name, pretrained (or not), number of channels,
    and number of outputs

    Inputs:
        name - string corresponding to model name
        pretrained- Boolean for whether a pretrained
                    model is requested
        num_channels- int number of channels
        num_classes- number of outputs of the network
    """
    function = getattr(models, name)
    model = function(pretrained=pretrained)

    if "resnet" in name:
        if num_channels == 1:
            model = ResNet18Grayscale(models.resnet.BasicBlock,
                                          [2, 2, 2, 2],
                                          num_classes)
        else:
            model.fc = nn.Linear(512, num_classes)
    else:
        model = nn.Sequential(*(list(model.children())[:-1]))
        model.classifier.add_module('6', nn.Linear(
                list(model.classifier.children()))[-3].in_features, num_classes)

    return model


class ResNet18Grayscale(models.resnet.ResNet):
    """
    A class that inherits the torchvision model
    Resnet and makes it compatible with grayscale
    images.
    """

    def __init__(self, block, layers, num_classes):
        super(ResNet18Grayscale, self).__init__(block, layers, num_classes)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.fc = nn.Linear(512, num_classes)
