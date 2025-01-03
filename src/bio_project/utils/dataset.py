import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset


class FeatureDataset(Dataset):
    def __init__(self, h5_file, top_k=100):
        self.features, self.labels, self.patch_names, self.cell_counts = self._load_data(h5_file, top_k)

    def _load_data(self, h5_file, top_k):
        with h5py.File(h5_file, 'r') as f:
            features = f['combined_features'][:]
            labels = f['combined_features'][:, -2]
            patch_names = f['patch_names'][:]
            cell_counts = f['combined_features'][:, -5]
            
        # Select top k patches based on cell counts
        top_indices = np.argsort(cell_counts)[-top_k:]
        return features[top_indices], labels[top_indices], patch_names[top_indices], cell_counts[top_indices]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.patch_names[idx], self.cell_counts[idx]
    