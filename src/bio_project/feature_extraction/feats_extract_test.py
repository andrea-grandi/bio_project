import os
import torch
import h5py
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import sys

sys.path.append('/Users/andreagrandi/Developer/bio_project/src/bio_project/feature_extraction/DINO')  # Sostituisci con il percorso corretto
from vision_transformer import vit_small


# Funzione per costruire e caricare il modello DINO
def load_dino_model(checkpoint_path):
    model = vit_small(patch_size=16)  # Modello ViT piccolo, patch size 16
    state_dict = torch.load(checkpoint_path, map_location='cpu')  # Carica i pesi
    model.load_state_dict(state_dict, strict=True)
    model.eval()  # Modalità di inferenza
    return model

# Funzione di preprocessing delle immagini
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Funzione principale
def main(checkpoint_path, dataset_path, csv_path, output_path):
    model = load_dino_model(checkpoint_path)

    metadata_df = pd.read_csv(csv_path)
    image_paths = [Path(dataset_path) / f"{img_id}.png" for img_id in metadata_df['image_id']]

    with h5py.File(output_path, 'w') as h5_file:
        metadata_group = h5_file.create_group('metadata')
        for col in metadata_df.columns:
            metadata_group.create_dataset(col, data=metadata_df[col].values, dtype='S' if metadata_df[col].dtype == 'object' else metadata_df[col].dtype)

        feature_group = h5_file.create_group('features')
        for i, image_path in enumerate(image_paths):
            print(f"Estraendo feature per: {image_path}")
            image_tensor = preprocess_image(image_path)
            with torch.no_grad():
                features = model(image_tensor).cpu().numpy().flatten()
            feature_group.create_dataset(f'patch_{i}', data=features)

    print(f"Feature extraction completata. Risultati salvati in: {output_path}")


if __name__ == "__main__":
    # Imposta i percorsi
    checkpoint_path = "/Users/andreagrandi/Developer/bio_project/weights/dino_camelyon17/checkpoint_20x.pth"
    dataset_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/dataset/camelyon17/512x512_patches"
    csv_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/dataset/camelyon17/processed_metadata.csv"
    output_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/feature_extraction/output_feats/dino_feats.h5"
    
    main(checkpoint_path, dataset_path, csv_path, output_path)