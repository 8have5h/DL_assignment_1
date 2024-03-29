# here we will make dataset for our assignment

# import necessary libraries like dataset, transforms, etc
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets
import os

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
path = "/home/bhavesh/Desktop/DL_assignment_1/Birds_25"


train_dataset = CustomDataset(path + '/train', transform=transform)
val_dataset = CustomDataset(path + '/val', transform=transform)
test_dataset = CustomDataset(path + '/test', transform=transform)


