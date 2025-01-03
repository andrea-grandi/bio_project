import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.buffermil.custom_buffermil import Buffermil
from utils.dataset import FeatureDataset


def train():
  h5_file = '/Users/andreagrandi/Developer/bio_project/src/bio_project/feature_extraction/output_feats/features.h5'
  top_k = 100
  input_dim = 2053
  buffer_size = 50
  epochs = 10
  batch_size = 16
  learning_rate = 0.001

  # Preparazione dei dati
  dataset = FeatureDataset(h5_file, top_k=top_k)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  # Inizializzazione del modello
  args = None  # Configurazione da fornire
  state_dict_weights = None
  model = Buffermil(args, state_dict_weights)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # Training
  for epoch in range(epochs):
      model.train()
      epoch_loss = 0

      for features, labels, patch_names, cell_counts in dataloader:
          features, labels = features.float().cuda(), labels.float().cuda()

          # Forward pass
          outputs = model(features)
          loss = criterion(outputs.squeeze(), labels)

          # Backward pass
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          epoch_loss += loss.item()

      print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

  # Salvataggio del modello
  torch.save(model.state_dict(), 'buffermil_model.pth')
  print("Training completato e modello salvato!")


