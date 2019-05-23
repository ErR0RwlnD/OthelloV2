import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.add_module('bn1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(
            num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('bn2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate,
                                           growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i+1), layer)


class OthelloDensenet(nn.Module):
    def __init__(self, game, dropout=0.3):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.dropout = dropout

        super(OthelloDensenet, self).__init__()
        self.features = nn.Sequential(OrderedDict(
            [('conv0', nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)),
             ('bn0', nn.BatchNorm2d(64)),
             ('relu0', nn.ReLU(inplace=True))]))
        self.features.add_module(
            'denseblock1', _DenseBlock(6, 64, 32, 4, dropout))
        self.features.add_module('downBn', nn.BatchNorm2d(256))
        self.features.add_module('downRelu', nn.ReLU(inplace=True))
        self.features.add_module('downConv', nn.Conv2d(
            256, 128, kernel_size=1, stride=1, bias=False))
        self.features.add_module(
            'denseblock2', _DenseBlock(6, 128, 64, 4, dropout))

        # use global average pooling instead of linear layers
        self.fc_bn0 = nn.BatchNorm2d(512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, self.action_size, bias=True)
        self.fc2 = nn.Linear(512, 1, bias=True)

    def forward(self, x):
        x = x.view(-1, 1, self.board_x, self.board_y)
        x = self.features(x)

        x = self.gap(self.fc_bn0(x))
        x = x.view(-1, 512)

        pi = self.fc1(x)
        v = self.fc2(x)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
