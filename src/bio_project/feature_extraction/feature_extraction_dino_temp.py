import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import argparse
from torchvision import transforms
from torch.nn.functional import normalize
from DINO.vision_transformer import VisionTransformer
from torch_geometric.data import Data

# Loading DINO model
def load_dino_model(checkpoint_path):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=0) 
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "model" in k}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Visual feature extraction
def extract_visual_features(model, image_path):
    with torch.no_grad():
        image_tensor = preprocess_image(image_path)
        feature_vector = model(image_tensor).squeeze().cpu().numpy()
        return normalize(torch.tensor(feature_vector), dim=0).numpy()

# Create DataBatch object
def create_data_batch(visual_vector, numerical_vector, patch_id):
    num_nodes = len(visual_vector)

    # Example structure for DataBatch
    data = Data(
        x=torch.tensor(visual_vector, dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),  # Replace with actual edges if available
        y=torch.tensor([0], dtype=torch.long),
        edge_index_filtered=torch.empty((2, 0), dtype=torch.long),  # Placeholder
        edge_index_2=torch.empty((2, 0), dtype=torch.long),         # Placeholder
        edge_index_3=torch.empty((2, 0), dtype=torch.long),         # Placeholder
        childof=torch.arange(num_nodes, dtype=torch.long),
        level=torch.zeros(num_nodes, dtype=torch.long),
        name=torch.tensor([patch_id], dtype=torch.long),
        patch_label=torch.zeros(num_nodes, dtype=torch.long),
        x_coord=torch.zeros(num_nodes, dtype=torch.float32),
        y_coord=torch.zeros(num_nodes, dtype=torch.float32),
        batch=torch.zeros(num_nodes, dtype=torch.long),
        ptr=torch.tensor([0, num_nodes], dtype=torch.long)
    )

    # Additional numerical features
    data.num_cells = torch.tensor([numerical_vector[0]], dtype=torch.float32)
    data.mean_cell_area = torch.tensor([numerical_vector[1]], dtype=torch.float32)
    data.cell_density = torch.tensor([numerical_vector[2]], dtype=torch.float32)

    return data

# Main function to process patches
def process_patches(model, image_dir, numerical_features_path, output_dir):
    numerical_features = pd.read_csv(numerical_features_path)
    os.makedirs(output_dir, exist_ok=True)

    for index, row in numerical_features.iterrows():
        patch_id = row['patch_id']
        image_path = os.path.join(image_dir, patch_id)

        if not os.path.exists(image_path):
            print(f"Image {patch_id} not found. Skipping.")
            continue

        # Extract visual features
        visual_vector = extract_visual_features(model, image_path)

        # Extract numerical features
        numerical_vector = row[['num_cells', 'mean_cell_area', 'cell_density']].values

        # Create DataBatch
        data_batch = create_data_batch(visual_vector, numerical_vector, patch_id)

        # Save DataBatch to .pt
        output_path = os.path.join(output_dir, f"{patch_id}.pt")
        torch.save(data_batch, output_path)

        print(f"Saved features for {patch_id} to {output_path}")

# Main script
def main():
    checkpoint_path = "/Users/andreagrandi/Developer/bio_project/weights/dino_camelyon17/checkpoint_20x.pth"
    image_dir = "/Users/andreagrandi/Developer/bio_project/src/bio_project/dataset/camelyon17/512x512_patches"
    numerical_features_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/dataset/camelyon17/cellpose_metadata.csv"
    output_dir = "/Users/andreagrandi/Developer/bio_project/src/bio_project/feature_extraction/output_feats/"

    arg = argparse.ArgumentParser()
    arg.add_argument("--checkpoint_path", type=str, default=checkpoint_path)
    arg.add_argument("--image_dir", type=str, default=image_dir)
    arg.add_argument("--numerical_features_path", type=str, default=numerical_features_path)
    arg.add_argument("--output_dir", type=str, default=output_dir)
    args = arg.parse_args()

    model = load_dino_model(args.checkpoint_path)

    print("Processing patches...")
    process_patches(model, args.image_dir, args.numerical_features_path, args.output_dir)
    print("Processing complete.")

if __name__ == "__main__":
    main()