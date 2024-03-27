import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from norm import NoNorm, BatchNorm, InstanceNorm, LayerNorm, GroupNorm, BatchInstanceNorm



norm1 = (0.4914, 0.4822, 0.4465)
norm2 = (0.2023, 0.1994, 0.2010)
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

transform = transforms.Compose([
    transforms.Resize((256, 256)), # resize to 256x256 pixels and 3 channels
    transforms.ToTensor(),
    transforms.Normalize(norm1, norm2)
])
path = os.getcwd()

train_dataset = CustomDataset(root_dir=path + '/train', transform=transform)
test_dataset = CustomDataset(root_dir=path + '/test', transform=transform)
val_dataset = CustomDataset(root_dir=path + '/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


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
    


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = ResNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-4,weight_decay=1e-5, momentum=0.9)
schedulers = [
    optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=- 1, verbose=False),
    optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, verbose=False)
    ]
scheduler =  schedulers[1] #Check for self.epochs param
criterion = nn.CrossEntropyLoss()


import json
from collections import defaultdict
from time import time
from tqdm import tqdm
from datetime import datetime

now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M")
def train(args, train_loader, val_loader):

    net = ResNet(norm_type = args['norm_type'])
    print(net)
    net = net.to(device)
    print("Model Loaded on Device:", device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    schedulers = [
        optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=- 1, verbose=False),
        optim.lr_scheduler.CosineAnnealingLR(optimizer, args['epochs'], verbose=False)
        ]
    scheduler =  schedulers[1] #Check for self.epochs param
    
    loss_tracker = defaultdict(list)
    accuracy_tracker = defaultdict(list)    
    time_tracker = defaultdict(list)
    ft_quantile_tracker = defaultdict(list)

    best_accuracy = -1
    best_accu_epoch = -1

    print("\n\n---------------------------- MODEL TRAINING BEGINS ----------------------------")
        
    t0 = time()
    for epoch in range(args['epochs']):
        print("\n#------------------ Epoch: %d ------------------#" % epoch)

        train_loss = []
        correct_pred = 0
        total_samples = 0
        
        net.train()
        for idx, batch in enumerate(train_loader):
            # print(idx, len(batch[0]))
            optimizer.zero_grad()
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            
            loss = criterion(outputs, labels)
            
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            _, pred = outputs.max(1)
            verdict = torch.eq(pred, labels)
            correct_pred += verdict.sum().item()
            total_samples += labels.size(0)
            if (idx == 10):
                break
            print("Batch: {}, Loss: {}, Accuracy: {}%".format(idx, loss.item(), round(correct_pred/total_samples*100, 2) ))

        loss_tracker["train"].append(np.mean(train_loss))
        accuracy_tracker["train"].append(round(correct_pred/total_samples*100, 2))

        scheduler.step()
        print("validating...")
        net.eval()
        correct_pred = 0
        total_samples = 0
        val_loss = []
        feature_list = []
        for idx, batch in enumerate(val_loader):
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)

            loss = criterion(outputs, labels)        
            val_loss.append(loss.item())

            _, pred = outputs.max(1)
            verdict = torch.eq(pred, labels)
            correct_pred += verdict.sum().item()
            total_samples += labels.size(0)
            if (idx == 10):
                break
            
            feature_list.extend(list(net.get_features().view(-1,1).numpy()))
        
        loss_tracker["val"].append(np.mean(val_loss))
        val_accuracy = round(correct_pred/total_samples*100, 2)
        accuracy_tracker["val"].append(val_accuracy)
        
        ft_quantile_tracker[1].append(np.percentile(feature_list, 1))
        ft_quantile_tracker[20].append(np.percentile(feature_list, 20))
        ft_quantile_tracker[80].append(np.percentile(feature_list, 80))
        ft_quantile_tracker[99].append(np.percentile(feature_list, 99))

        t1 = time()

        print("Epoch: {}, Total Time Elapsed: {}Mins, Train Loss: {}, Train Accuracy: {}%, Validation Loss: {}, Validation Accuracy: {}%".format(epoch, round((t1-t0)/60,2), loss_tracker["train"][-1], accuracy_tracker["train"][-1], loss_tracker["val"][-1], accuracy_tracker["val"][-1]))
        time_tracker['train'].append(round((t1-t0)/60,2))
        model_state = {
                'accu': val_accuracy,
                'epoch': epoch,
                'best_accu': best_accuracy,
                'best_accu_epoch': best_accu_epoch
            }

        print("Epoch: {}, Saving Model Checkpoint: {}".format(epoch, now.strftime("%d-%m-%y %H:%M")))
        
        torch.save(net, os.path.join(args['checkpoint_dir'] , "latest_checkpoint_{}.pth".format(args['norm_type'])))
        with open(os.path.join(args['checkpoint_dir'], "training_progress_{}.json".format(args['norm_type'])), "w") as outfile:
            json.dump(model_state, outfile)
        
        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy
            best_accu_epoch = epoch

            model_state = {
                'accu': val_accuracy,
                'epoch': epoch,
                'best_accu': best_accuracy,
                'best_accu_epoch': best_accu_epoch
            }
            
            print("Best Validation Accuracy Updated = {}%, Last Best = {}%".format(val_accuracy, best_accuracy))
            print("Saving Best Model Checkpoint:", now.strftime("%d-%m-%y %H:%M"))

            torch.save(net, os.path.join(args['checkpoint_dir'] , "best_val_checkpoint_{}.pth".format(args['norm_type'])))
            with open(os.path.join(args['checkpoint_dir'] , "training_progress_{}.json".format(args['norm_type'])), "w") as outfile:
                json.dump(model_state, outfile)


        with open(os.path.join(args['result_dir'],"loss_tracker_{}_{}.json".format(args['norm_type'] , date_time)), "w") as outfile:
            json.dump(loss_tracker, outfile)

        with open(os.path.join(args['result_dir'],"accuracy_tracker_{}_{}.json".format(args['norm_type'] , date_time)), "w") as outfile:
            json.dump(accuracy_tracker, outfile)

        with open(os.path.join(args['result_dir'],"time_tracker_{}_{}.json".format(args['norm_type'] , date_time)), "w") as outfile:
            json.dump(time_tracker, outfile)

        with open(os.path.join(args['result_dir'], "ft_quantile_tracker_{}_{}.json".format(args['norm_type'] , date_time)), "w") as outfile:
            json.dump(ft_quantile_tracker, outfile)
    return


if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("results"):
        os.makedirs("results")
    
    norm_type_mappings = {"torch_bn": "Pytorch Batch Normalization", "nn": "No Normalization", 
    "bn": "Custom Batch Normalization", "in": "Instance Normalization","ln": "Layer Normalization", "bin": "Batch Instance Normalization",
    "gn": "Group Normalization"}
    for key in norm_type_mappings:
        if not os.path.exists(os.path.join("checkpoints", key)):
            os.makedirs(os.path.join("checkpoints", key))
        if not os.path.exists(os.path.join("results", key)):
            os.makedirs(os.path.join("results", key))

    for norm_type in norm_type_mappings:
        args = {
            'epochs': 10,
            'norm_type': norm_type,
            'checkpoint_dir': os.path.join("checkpoints", norm_type),
            'result_dir': os.path.join("results", norm_type)
        }
        train(args, train_loader, val_loader)
