import pandas as pd
import os
import shutil

def find_patch_in_subdirectories(base_dir, patch_filename):
    for root, _, files in os.walk(base_dir):
        if patch_filename in files:
            return os.path.join(root, patch_filename)
    return None


input_metadata_file = "camelyon17_dataset_10000_instances/metadata.csv"
output_metadata_file = "selected_metadata.csv"
original_patches_dir = "camelyon17_dataset_10000_instances/patches"
output_patches_dir = "selected_patches"

n_patches = 1000

metadata = pd.read_csv(input_metadata_file)

required_columns = ["patient", "node", "x_coord", "y_coord", "tumor", "slide", "center", "split"]
for column in required_columns:
    if column not in metadata.columns:
        raise ValueError(f"La colonna '{column}' non esiste nel file dei metadati.")

selected_patches = metadata.sample(n=n_patches, random_state=42)
selected_patches.to_csv(output_metadata_file, index=False)
os.makedirs(output_patches_dir, exist_ok=True)

for _, row in selected_patches.iterrows():
    patient_str = f"{int(row['patient']):03d}" 
    patch_filename = f"patch_patient_{patient_str}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png"
    source_path = find_patch_in_subdirectories(original_patches_dir, patch_filename)
    destination_path = os.path.join(output_patches_dir, patch_filename)

    if source_path:
        shutil.copy(source_path, destination_path)
    else:
        print(f"Attenzione: la patch '{patch_filename}' non esiste nelle sottocartelle di '{original_patches_dir}'.")

print(f"Selezionate {n_patches} patch casuali e copiate in '{output_patches_dir}'. Metadati salvati in '{output_metadata_file}'.")