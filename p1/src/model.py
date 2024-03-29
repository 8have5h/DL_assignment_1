import torch
import torch.nn as nn
import torch.nn.functional as F
from norm import *


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, normalization_type = "torch_bn"):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = get_normalization(out_channels, normalization_type)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = get_normalization(out_channels, normalization_type)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return 
        

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, normalization_type = "torch_bn"):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = get_normalization(16, normalization_type)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0], normalization_type)
        self.layer2 = self.make_layer(block, 32, layers[1], normalization_type, 2)
        self.layer3 = self.make_layer(block, 64, layers[2], normalization_type, 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, normalization_type, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                get_normalization(out_channels, normalization_type),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, normalization_type))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, normalization_type=normalization_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    