import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# class ResidualBlock(nn.Module):

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__() # super() function makes class inheritance
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False) # padding = 1, so that the size of the input and output is the same
        self.bn1 = nn.BatchNorm2d(out_channels) # batch normalization 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False) # padding = 1, so that the size of the input and output is the same
        self.bn2 = nn.BatchNorm2d(out_channels) # batch normalization
        if stride != 1 or in_channels != out_channels: # if the size of the input and output is different
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ) # make the size of the input and output the same

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # ReLU activation function
        out = self.bn2(self.conv2(out)) # batch normalization
        out += x # identity
        out = F.relu(out) # ReLU activation function
        return out 

class ResNet(nn.Module):
    def __init__(self, block, n, r=25):
        super(ResNet, self).__init__() 
        self.in_channels = 16 # initial input channel -> channel means the number of filters
        
        # First conv layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False) # initial input channel = 3
        # padding = 1, so that the size of the input and output is the same
        self.bn1 = nn.BatchNorm2d(16) # batch normalization
        
        # Layers for different block sizes
        self.layer1 = self._make_layer(block, 16, n, stride=1) # 16 channels
        self.layer2 = self._make_layer(block, 32, n, stride=2) # 32 channels
        self.layer3 = self._make_layer(block, 64, n, stride=2) # 64 channels
        
        # Final fully connected layer
        self.linear = nn.Linear(64, r) # 64 channels

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # this function makes the layers, which is used in the __init__ function
        # block: ResidualBlock, out_channels: 16, 32, 64, num_blocks: 2, 2, 2, stride: 1, 2, 2
        strides = [stride] + [1]*(num_blocks-1) # make the size of the input and output the same
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1)) # average pooling over all the spatial dimensions
        out = out.view(out.size(0), -1) # flatten the tensor
        out = self.linear(out) # fully connected layer
        # use softmax function for the output layer
        out = F.softmax(out, dim=1)
        return out
