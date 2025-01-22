import pandas as pd
from sklearn.model_selection import train_test_split

# Path to the uploaded CSV file
input_csv_path = '/Users/andreagrandi/Developer/bio_project/src/bio_project/dataset/camelyon17/selected_metadata.csv'

# Load the CSV file
metadata = pd.read_csv(input_csv_path)

# Add a new column 'image' with the formatted file names
metadata['image'] = metadata.apply(
    lambda row: f"patch_patient_{str(row['patient']).zfill(3)}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png",
    axis=1
)

# Rename the tumor column to 'label' and ensure it's binary (0 or 1)
metadata['label'] = metadata['tumor'].astype(int)

# Perform a train-test split (80% train, 20% test)
train, test = train_test_split(metadata, test_size=0.2, random_state=42, stratify=metadata['label'])

# Add the 'phase' column indicating train or test
train['phase'] = 'train'
test['phase'] = 'test'

# Combine train and test dataframes
final_metadata = pd.concat([train, test], axis=0, ignore_index=True)

# Select only the required columns
final_metadata = final_metadata[['image', 'label', 'phase']]

# Save the new CSV file
output_csv_path = '/Users/andreagrandi/Developer/bio_project/src/bio_project/dataset/camelyon17/dino_feats_extractor_metadata.csv'
final_metadata.to_csv(output_csv_path, index=False)
