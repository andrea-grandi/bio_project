import argparse
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import h5py
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataset(Dataset):
    def __init__(self, patches_dir, csv_path, transform=None):
        self.patches_list = sorted(glob.glob(os.path.join(patches_dir, '**', '*.jpg'), recursive=True))
        self.features_df = pd.read_csv(csv_path)
        self.transform = transform

        # Create mapping from patch_id to numerical features
        self.id_mapping = self.create_id_mapping()

    def create_id_mapping(self):
        """Create a mapping from image filenames to their numerical features."""
        mapping = {}
        for _, row in self.features_df.iterrows():
            patch_id = row['patch_id'] 
            mapping[patch_id] = row.drop(['patch_id']).values.astype(np.float32)
        return mapping

    def __len__(self):
        return len(self.patches_list)

    def __getitem__(self, idx):
        # Load and transform image
        img_path = self.patches_list[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        # Extract patch metadata
        patch_id = os.path.basename(img_path)
        
        # Get corresponding numerical features using mapping
        if patch_id not in self.id_mapping:
            raise ValueError(f"No numerical features found for patch ID: {patch_id}")

        numerical_features = self.id_mapping[patch_id]
        #print(numerical_features)
        
        return img, numerical_features, patch_id


def get_feature_extractor():
    """Initialize ResNet50 model for feature extraction."""
    model = models.resnet50(pretrained=True)
    # Remove the last fully connected layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor

def extract_features(args):
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    # Initialize dataset and dataloader
    dataset = MyDataset(args.patches_dir, args.csv_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=args.num_workers)

    # Initialize feature extractor
    feature_extractor = get_feature_extractor()

    # Prepare output file
    output_file = os.path.join(args.output_dir, 'features.h5')
    
    # Extract and save features
    all_features = []
    all_combined_features = []
    patch_names = []
    
    with torch.no_grad():
        for batch, numerical_feats, names in dataloader:
            # Extract visual features
            batch = batch.to(device)
            visual_features = feature_extractor(batch).squeeze()
            
            # Combine with numerical features
            visual_features = visual_features.cpu().numpy()
            numerical_feats = numerical_feats.numpy()
            
            # Concatenate features
            combined_features = np.concatenate([visual_features, numerical_feats], axis=1)
            
            all_features.extend(visual_features)
            all_combined_features.extend(combined_features)
            patch_names.extend(names)

    # Save to H5 file
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('features', data=np.array(all_features))
        f.create_dataset('combined_features', data=np.array(all_combined_features))
        f.create_dataset('patch_names', data=np.array(patch_names, dtype='S'))

def main():
    parser = argparse.ArgumentParser(description='Feature extraction for WSI patches')
    parser.add_argument('--patches_dir', type=str, required=True,
                        help='Directory containing image patches')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file with numerical features')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save extracted features')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    extract_features(args)


if __name__ == '__main__':
    main()