import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
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
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
path = os.getcwd()

train_dataset = CustomDataset(root_dir=path + '/train', transform=transform)
test_dataset = CustomDataset(root_dir=path + '/test', transform=transform)
val_dataset = CustomDataset(root_dir=path + '/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


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

        self.shortcut = nn.Sequential() # identity
        if stride != 1 or in_channels != out_channels: # if the size of the input and output is different
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ) # make the size of the input and output the same

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # ReLU activation function
        out = self.bn2(self.conv2(out)) # batch normalization
        out += self.shortcut(x) # identity
        out = F.relu(out) # ReLU activation function
        return out 

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=25):
        super(ResNet, self).__init__() 
        self.in_channels = 16 # initial input channel
        
        # First conv layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False) # initial input channel = 3
        self.bn1 = nn.BatchNorm2d(16) # batch normalization
        
        # Layers for different block sizes
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1) # 16 channels
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2) # 32 channels
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2) # 64 channels
        
        # Final fully connected layer
        self.linear = nn.Linear(64, num_classes) # 64 channels

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
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(device)
model = ResNet(ResidualBlock, [2, 2, 2], num_classes=25).to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train(model, device, train_loader, optimizer, epoch):
    # if model exists, load model
    if os.path.exists(f"model_{epoch-1}.pt"):
        model.load_state_dict(torch.load(f"model_{epoch-1}.pt"))
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss {loss.item()}')
        # if keyboard interrupt, save model
        if batch_idx % 20 == 0:
            torch.save(model.state_dict(), f"model_{epoch}.pt")


def test(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    targets, preds = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            targets.extend(target.view_as(pred).cpu().numpy())
            preds.extend(pred.cpu().numpy())

    test_loss /= len(loader.dataset)
    accuracy = accuracy_score(targets, preds)
    micro_f1 = f1_score(targets, preds, average='micro')
    macro_f1 = f1_score(targets, preds, average='macro')
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}\n')
    return test_loss, accuracy, micro_f1, macro_f1

# for epoch in range(1, 51):
#     train(model, device, train_loader, optimizer, epoch)
#     test(model, device, train_loader)
#     test(model, device, val_loader)
#     scheduler.step()
import matplotlib.pyplot as plt

# Initialize lists to store metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
train_micro_f1s, val_micro_f1s = [], []
train_macro_f1s, val_macro_f1s = [], []

for epoch in range(1, 51):
    train(model, device, train_loader, optimizer, epoch)
    # save model
    torch.save(model.state_dict(), f"model_{epoch}.pt")
    train_loss, train_accuracy, train_micro_f1, train_macro_f1 = test(model, device, train_loader)
    val_loss, val_accuracy, val_micro_f1, val_macro_f1 = test(model, device, val_loader)
    scheduler.step()

    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    train_micro_f1s.append(train_micro_f1)
    val_micro_f1s.append(val_micro_f1)
    train_macro_f1s.append(train_macro_f1)
    val_macro_f1s.append(val_macro_f1)

# Function to plot metrics
def plot_metrics(train_metrics, val_metrics, title, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label='Train')
    plt.plot(val_metrics, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

# Plot each metric
plot_metrics(train_losses, val_losses, 'Loss Over Epochs', 'Loss')
plot_metrics(train_accuracies, val_accuracies, 'Accuracy Over Epochs', 'Accuracy')
plot_metrics(train_micro_f1s, val_micro_f1s, 'Micro F1 Score Over Epochs', 'Micro F1 Score')
plot_metrics(train_macro_f1s, val_macro_f1s, 'Macro F1 Score Over Epochs', 'Macro F1 Score')
