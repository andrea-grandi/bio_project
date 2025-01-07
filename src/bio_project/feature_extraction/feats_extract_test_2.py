import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import h5py
import argparse

from torchvision import transforms
from torch.nn.functional import normalize

from DINO.vision_transformer import VisionTransformer 


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
def extract_visual_features(model, image_dir):
    features = {}
    with torch.no_grad():
        for image_name in os.listdir(image_dir):
            if image_name.endswith(".png"):
                image_path = os.path.join(image_dir, image_name)
                image_tensor = preprocess_image(image_path)
                feature_vector = model(image_tensor).squeeze().cpu().numpy()
                feature_vector = normalize(torch.tensor(feature_vector), dim=0).numpy()
                features[image_name] = feature_vector

                # Debug
                print(f"Feature extracted for {image_name}")
                #print(f"Feature vector: {feature_vector}")

    return features

# Feature concatenation
def concatenate_features(visual_features, numerical_features_path):
    numerical_features = pd.read_csv(numerical_features_path)
    concatenated_features = {}

    for index, row in numerical_features.iterrows():
        patch_id = row['patch_id']
        if patch_id in visual_features:
            visual_vector = visual_features[patch_id]
            numerical_vector = row[['num_cells', 'mean_cell_area', 'cell_density']].values
            concatenated_vector = np.concatenate([visual_vector, numerical_vector])
            concatenated_features[patch_id] = concatenated_vector

    return concatenated_features

# Saving features
def save_features_to_pt(features, output_path):
    torch.save(features, output_path)


def main():
    checkpoint_path = "/Users/andreagrandi/Developer/bio_project/weights/dino_camelyon17/checkpoint_20x.pth"
    image_dir = "/Users/andreagrandi/Developer/bio_project/src/bio_project/dataset/camelyon17/512x512_patches"
    numerical_features_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/dataset/camelyon17/cellpose_metadata.csv"
    output_pt_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/feature_extraction/output_feats/concatenated_features.pt"

    arg = argparse.ArgumentParser()
    arg.add_argument("--checkpoint_path", type=str, default=checkpoint_path)
    arg.add_argument("--image_dir", type=str, default=image_dir)
    arg.add_argument("--numerical_features_path", type=str, default=numerical_features_path)
    arg.add_argument("--output_pt_path", type=str, default=output_pt_path)
    args = arg.parse_args()

    model = load_dino_model(args.checkpoint_path)

    # Visual feature extraction
    print("Visual feature extraction...")
    visual_features = extract_visual_features(model, args.image_dir)

    # Feature concatenation
    print("Feature concatenation...")
    concatenated_features = concatenate_features(visual_features, args.numerical_features_path)

    # Saving features 
    print("Saving features to pt...")
    save_features_to_pt(concatenated_features, args.output_pt_path)
    print(f"Feature saved in {args.output_pt_path}")


if __name__ == "__main__":
    main()
