import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os


class TransformerMIL(nn.module):
  def __init__(self, num_molecular_features, num_classes=2, d_model=512, nhead=8, num_layers=6):
    super(TransformerMIL, self).__init__()

    self.patch_embedding = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
    self.positional_encoding = nn.Parameter(torch.randn(1, 196, d_model))  # 196 = (224/16) * (224/16)
    
    self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
    
    self.molecular_embedding = nn.Linear(num_molecular_features, d_model)
    
    self.attention_layer = nn.Sequential(
        nn.Linear(d_model, 128),
        nn.Tanh(),
        nn.Linear(128, 1)
    )
    
    self.classifier = nn.Linear(d_model, num_classes)

def forward(self, images, molecular_features):
    patches = self.patch_embedding(images)  # [batch_size, d_model, 14, 14]
    patches = patches.flatten(2)  # [batch_size, d_model, 196]
    patches = patches.permute(0, 2, 1)  # [batch_size, 196, d_model]
    
    # Positional Encoding
    patches = patches + self.positional_encoding
    
    molecular_embedding = self.molecular_embedding(molecular_features)
    molecular_embedding = molecular_embedding.unsqueeze(1)  # Aggiungi una dimensione per concatenazione
    
    features = torch.cat([patches, molecular_embedding], dim=1)
    
    transformer_output = self.transformer(features, features)
    
    attention_weights = torch.softmax(self.attention_layer(transformer_output), dim=1)
    mil_representation = torch.sum(attention_weights * transformer_output, dim=1)
    
    output = self.classifier(mil_representation)
    return output