import torch
import pandas as pd
from torchvision.models import resnet50
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

# Pretrained ResNet50
resnet = resnet50(pretrained=True)
# Remove last layer
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]) 
resnet.eval()

# Metadata
metadata = pd.read_csv('selected_patches_from_camelyon17/metadata.csv')
extracted_metadata = pd.read_csv('selected_patches_from_camelyon17/extracted_metadata.csv')

# Patches
image_patch_path = "selected_patches_from_camelyon17/"

# Normalization of numerical features
for col in extracted_metadata.columns[1:]:
    extracted_metadata[col] = (extracted_metadata[col] - extracted_metadata[col].mean()) / extracted_metadata[col].std()

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
	image_filename = f"patch_patient_{str(row['patient']).zfill(3)}_node_{row['node']}_x_{row['x']}_y_{row['y']}.png"
	print(image_filename)
    image_path = f"{image_patch_path}/{image_filename}"
    visual_features = extract_features(image_path)
    numerical_features = torch.tensor(extracted_metadata.loc[idx, extracted_metadata.columns[1:]].values, dtype=torch.float32)
    combined = torch.cat((visual_features, numerical_features))
    combined_features.append(combined)

# to tensor
combined_features = torch.stack(combined_features)