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
from resnet import ResNet
from train import train



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




norms = ["torch_bn", "bn", "nn", "in", "ln", "gn", "bin"]

for norm_type in norms:
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("logs"):
        os.makedirs("logs")
    for folder in ["checkpoints", "results", "logs"]:
        if not os.path.exists(folder + "/" + norm_type):
            os.makedirs(folder + "/" + norm_type)
    print("\n Training with Norm Type:", norm_type)
    args = {
        "norm_type": norm_type,
        "epochs": 50,
        "checkpoint_dir": f"checkpoints/{norm_type}",
        "log_dir": f"logs/{norm_type}",
        "result_dir": f"results/{norm_type}"
    }

    train(args, train_loader, val_loader)
    print("\n\n\n\n")

