import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models.buffermil import custom_buffermil as Buffermil


class CustomDataset(Dataset):
    def __init__(self, patch_dir, feature_file, transform=None):
        self.patch_dir = patch_dir
        self.feature_dir = feature_dir
        self.transform = transform

        self.patches = sorted([f for f in os.listdir(patch_dir) if f.endswith('.png')])

        data = torch.load(feature_file)
        self.patch_ids = sorted(data.keys())

        self.mapping = {
            os.path.splitext(patch)[0]: idx for idx, patch in enumerate(self.patches)
        }

        #print(type(data))
        #print(data.keys())

        if not self.mapping:
            raise ValueError("Error: No patches found in the dataset")


    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_path = os.path.join(self.patch_dir, self.patches[idx])
        feature_path = os.path.join(self.feature_dir, self.features[idx])

        # Load patch image
        patch = Image.open(patch_path).convert("RGB")
        if self.transform:
            patch = self.transform(patch)

        # Load features and metadata
        data = torch.load(feature_path)
        visual_features = data["visual_features"]
        cell_metadata = {
            "cell_count": data["cell_count"],
            "mean_cell_area": data["mean_cell_area"],
            "cell_density": data["cell_density"]
        }

        label = data["label"] 

        return patch, visual_features, cell_metadata, label

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for patches, features, metadata, labels in dataloader:
            patches = patches.to(device)
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(features, None, None, None) 
            loss = criterion(outputs["higher"], labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")

    print("Training complete!")
    return model

if __name__ == "__main__":
    patch_dir = "/Users/andreagrandi/Developer/bio_project/src/bio_project/dataset/camelyon17/512x512_patches"
    feature_dir = "/Users/andreagrandi/Developer/bio_project/src/bio_project/feature_extraction/output_feats/cellpose_concatenated_features.pt"

    batch_size = 16
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = CustomDataset(patch_dir, feature_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    args = {
        "bufferaggregate": "mean",
        "buffer_freq": 10,
        "randomstore": False,
        "ntop": 10
    }
    state_dict_weights = None
    model = Buffermil(args, state_dict_weights).to(device)

    # Loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    model = train_model(model, dataloader, criterion, optimizer, device, num_epochs=num_epochs)
