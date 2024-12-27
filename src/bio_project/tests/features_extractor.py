import torch
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import ast

# Pretrained ResNet50
weights = ResNet50_Weights.IMAGENET1K_V1 
model = resnet50(weights=weights)
# Remove last layer
resnet = torch.nn.Sequential(*list(model.children())[:-1]) 
resnet.eval()

# Metadata
metadata = pd.read_csv('selected_patches_from_camelyon17/final_metadata.csv')

# Patches
image_patch_path = "selected_patches_from_camelyon17/"

def normalize():
    # Normalization of numerical features
    for col in metadata.columns[1:]:
        metadata[col] = (metadata[col] - metadata[col].mean()) / metadata[col].std()

# Pipeline of image transformation
transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_tensor).squeeze().view(-1)
    return features

# Combine visual features and numerical ones
combined_features = []
for idx, row in metadata.iterrows():
    image_filename = f"patch_patient_{str(int(row['patient'])).zfill(3)}_node_{int(row['node'])}_x_{int(row['x_coord'])}_y_{int(row['y_coord'])}.png"
    image_path = f"{image_patch_path}{image_filename}"
    normalize()
    visual_features = extract_features(image_path)
    if idx in metadata.index:
        numerical_features = torch.tensor(
            metadata.loc[idx, metadata.columns[1:]].values, dtype=torch.float32
        )
    else:
        print(f"Indice {idx} non trovato in metadata.")

    # concat visual features with numerical features
    combined = torch.cat((visual_features, numerical_features))
    combined_features.append(combined)

# to tensor
combined_features = torch.stack(combined_features)
print(combined_features.shape) # (1000, 2058)
print(combined_features[0]) 
print("Done")
