import torch 
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from google.colab import drive
#drive.mount('/content/drive')

print('Hello World')

import kagglehub


#'/Users/alexrendell/Library/CloudStorage/GoogleDrive-alex.g.rendell@gmail.com/My Drive'

# Load the dataset from Kaggle

# Download latest version
#path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images", path="/Users/alexrendell/Library/CloudStorage/GoogleDrive-alex.g.rendell@gmail.com/My Drive")

#print("Path to dataset files:", path)


#import kagglehub

# Download latest version
#path = kagglehub.dataset_download("ambarish/breakhis")

#print("Path to dataset files:", path) # /Users/alexrendell/.cache/kagglehub/datasets/ambarish/breakhis/versions/4

from Data_Preprocessing import BreastCancerDataset

dataset = ""

new = BreastCancerDataset(root_dir=dataset, transform=None)

