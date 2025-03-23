import numpy as np
import matplotlib.pyplot as plt

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from PIL import Image

# Define transformations:
# Resixe images to 224 x 224 (A common size for CNN's)
# Convert images to PyTorch tensors
# Normalise pixel values to [-1,1]
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1]
])

class BreastCancerDataset(Dataset):
    """A custom dataset for loading and labeling breast cancer images."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [] # Sotres image file paths
        self.labels = [] # Store labels (0 = Benign, 1 = Malignant)
        
        # Loop through all files in the dataset directory
        for img_name in os.listdir(self.root_dir):
            if img_name.endswith('png'): # Only want to process .png files (images)
                img_path = os.path.join(root_dir, img_name)
                self.image_paths.append(img_path)
                
                if 'SOB_M' in img_name:
                    self.labels.append(1)  # Malignant
                else:
                    self.labels.append(0)  # Benign
                    
        print(f"Image Paths: {self.image_paths}")
    def __len__(self):
        """Return the total number of samples"""
        
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """Return a sample from the dataset"""
        
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        
        # If the image hasn't been transformed, then transform it
        if self.transform:
            image = self.transform(image)
        
        return image, label
                    
    #print("Class BreastCancerDataset is defined correctly")
    

#database_dir = "/Users/alexrendell/Documents/MSc - Advanced Computer Science/CSCM10-Computer_Science_Project_Research_Methods/Databases/BreakHis_BreastCancer"

#dataset = BreastCancerDataset(database_dir, transform=transform)

#print(len(dataset.labels))