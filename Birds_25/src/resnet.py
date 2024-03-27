import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from norm import NoNorm, BatchNorm, InstanceNorm, LayerNorm, GroupNorm, BatchInstanceNorm

def layer_normalization(dim, norm_type):
    if norm_type == "torch_bn" or norm_type == "inbuilt":
        return nn.BatchNorm2d(dim)

    elif norm_type == "bn":
        return BatchNorm(num_features=dim)

    elif norm_type == "nn":
        return NoNorm()

    elif norm_type == "in":
        return InstanceNorm(num_features=dim)

    elif norm_type == "ln":
        return LayerNorm(num_features=dim)
    
    elif norm_type == "gn":
        return GroupNorm(num_features=dim)

    elif norm_type == "bin":
        return BatchInstanceNorm(num_features=dim)

    else:
        pass

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None, norm_type="torch_bn"):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
            padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, 
            padding=1, bias=False
        )
        self.relu = nn.ReLU()
        self.norm1 = layer_normalization(out_channels, norm_type)
        self.norm2 = layer_normalization(out_channels, norm_type)

    def forward(self, x):
        x_residual = x
        
        if self.downsample is not None:
            x_residual = self.downsample(x) # downsample the input x to match the output dimensions
            
        out = self.norm1(self.conv1(x)) # apply the first convolutional layer
        out = self.relu(out)
        out = self.norm2(self.conv2(out))
        
        out += x_residual
        out = self.relu(out)
        
        return out
    

class ResNet(nn.Module):
    def __init__(self, n_channels = [16, 32, 64], n_layers = [2, 2, 2], n_classes = 25, norm_type = "torch_bn"):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False) # 3 input channels for RGB
        self.layer_norm = layer_normalization(n_channels[0], norm_type)
        self.relu = nn.ReLU()
        self.in_channels = n_channels[0]        
        self.out_channels = 0
        self.features = None
                 
        layers = {}
        for c in range(len(n_channels)):
            layer = []
            self.out_channels = n_channels[c]
            n = n_layers[c]
            
            for l in range(n):
                downsample = None                
                if self.in_channels != self.out_channels:
                    """CHECK KERNEL SIZE HERE"""
                    downsample = nn.Sequential(
                        nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, 
                                  stride=2, padding=1, bias=False), 
                        layer_normalization(self.out_channels, norm_type)
                    )
                if c > 0 and l == 0:
                    stride = 2
                else:
                    stride = 1
                layer.append(ResidualBlock(self.in_channels, self.out_channels, stride = stride, downsample = downsample, norm_type = norm_type))
                if l == 0:
                    self.in_channels = self.out_channels       
            layers[c+1] = layer
            
        self.layer1 = nn.Sequential(*layers[1]) # * unpacks the list, so it's like passing each element of the list as an argument
        self.layer2 = nn.Sequential(*layers[2])
        self.layer3 = nn.Sequential(*layers[3])




        # do a mean pool
        self.avg_pool = nn.AvgPool2d(64)
        self.fc = nn.Linear(64, n_classes)
        
    def forward(self, x):
        # print("Input Shape:", x.shape)
        # input convolution
        x = self.layer_norm(self.conv(x))
        x = self.relu(x)
        # print("first conv:", x.shape)
        # residual layers
        x = self.layer1(x)
        # print("layer 1 done:", x.shape)
        x = self.layer2(x)
        # print("layer 2 done:", x.shape)
        x = self.layer3(x)
        # print("layer 3 done:", x.shape)
        
        # average pool
        x = self.avg_pool(x)
        
        # flatten and fc out
        self.features = x.view(-1).detach().cpu()
        x = x.view(-1, 64)
        x = self.fc(x)
        # print("fc done:", x.shape)
        return x

    def get_features(self):
        return self.features
    