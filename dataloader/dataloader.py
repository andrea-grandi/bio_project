import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os


class CancerDataset(Dataset):
    def __init__(self, image_dir, molecular_data_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.molecular_data = pd.read_csv(molecular_data_path)
        self.patient_ids = self.molecular_data['patient_id'].tolist()
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        image_path = os.path.join(self.image_dir, f"{patient_id}.png")
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        molecular_features = self.molecular_data.loc[self.molecular_data['patient_id'] == patient_id].drop('patient_id', axis=1).values
        molecular_features = torch.tensor(molecular_features, dtype=torch.float32)
        
        return image, molecular_features

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_dir = 'path/to/images'
molecular_data_path = 'path/to/molecular_data.csv'
dataset = CancerDataset(image_dir, molecular_data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
