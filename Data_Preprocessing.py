import numpy as np
import matplotlib.pyplot as plt

import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Define transformations (resize, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1]
])

class BreastCancerDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for img_name in os.listdir(self.root_dir):
            if img_name.endswith('png'):
                img_path = os.path.join(root_dir, img_name)
                self.image_paths.append(img_path)
                
                if 'SOB_M' in img_name:
                    self.labels.append(1)  # Malignant
                else:
                    self.labels.append(0)  # Benign
                    
    print("Class BreastCancerDataset is defined correctly")
    

database_dir = "/Users/alexrendell/Documents/MSc - Advanced Computer Science/CSCM10-Computer_Science_Project_Research_Methods/Databases/BreakHis_BreastCancer"

dataset = BreastCancerDataset(database_dir, transform=transform)

print(len(dataset.labels))