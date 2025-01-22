import os
import pandas as pd
import shutil
import argparse
from sklearn.model_selection import train_test_split


def split_dataset(output_dir, metadata, images_dir):
    splits = ["train", "val"]
    classes = ["tumor", "non_tumor"]

    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    train, val = train_test_split(metadata, test_size=0.2, stratify=metadata["tumor"], random_state=42)

    return train, val

def move_files(split_data, split_name, images_dir, output_dir):
    for _, row in split_data.iterrows():
        image_name = f"patch_patient_{str(row['patient']).zfill(3)}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png"
        label = "tumor" if row["tumor"] == 1 else "non_tumor"
        src_path = os.path.join(images_dir, image_name)
        dst_path = os.path.join(output_dir, split_name, label, image_name)
        print(f"Sposto {src_path} in {dst_path}")
        print(image_name)
        
        if os.path.exists(src_path):  # Verifica che l'immagine esista
            shutil.copy(src_path, dst_path)


def main():
    # Default paths
    images_dir = "/Users/andreagrandi/Developer/bio_project/src/bio_project/dataset/camelyon17/512x512_patches"  
    metadata_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/dataset/camelyon17/selected_metadata.csv"  
    output_dir = "/Users/andreagrandi/Developer/bio_project/src/bio_project/dataset/camelyon17/output_dataset"
  
    arg = argparse.ArgumentParser()
    arg.add_argument("--images_dir", type=str, default=images_dir)
    arg.add_argument("--metadata_path", type=str, default=metadata_path)
    arg.add_argument("--output_dir", type=str, default=output_dir)
    args = arg.parse_args()

    metadata = pd.read_csv(args.metadata_path) 

    train, val = split_dataset(args.output_dir, metadata, args.images_dir)

    move_files(train, "train", args.images_dir, args.output_dir)
    move_files(val, "val", args.images_dir, args.output_dir)

    print("DONE!")


if __name__ == "__main__":
    main()

