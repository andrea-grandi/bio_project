import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import cv2
from cellpose import models, io
import matplotlib.pyplot as plt


def visualize_segmentation(img, masks):
    centroids = []
    for i in range(1, np.max(masks) + 1):
        mask = (masks == i).astype(np.uint8)
        moments = cv2.moments(mask)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            centroids.append((cx, cy))

    contours, _ = cv2.findContours(masks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img_rgb, contours, -1, (0, 255, 0), 1)

    for cx, cy in centroids:
        cv2.circle(img_rgb, (cx, cy), 3, (0, 0, 255), -1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Segmentazione con Cellpose")
    plt.show()

def process_patches(input_dir, output_csv, channels=None, diameter=None, model_type="cyto3"):
    model = models.Cellpose(gpu=False, model_type=model_type)
    metadata = []
    # Recursively find all .jpg files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                patch_path = os.path.join(root, file)

                img = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
                print(f"Processing {file}...")

                # Auto-estimate diameter if not provided
                if diameter is None:
                    estimated_diameter = model.eval(img, channels=channels, diameter=None)[-1]
                    print(f"Estimated diameter: {estimated_diameter}")
                    diameter = estimated_diameter
                
                try:
                    masks, flows, styles, diams = model.eval(img, channels=channels, diameter=diameter)
                
                except Exception as e:
                    print(f"Segmentation Error: {e}")

                # Extract metadata
                num_cells = np.max(masks)  # Number of segmented cells
                cell_areas = [np.sum(masks == i) for i in range(1, num_cells + 1)]
                mean_cell_area = np.mean(cell_areas) if num_cells > 0 else 0
                cell_density = num_cells / img.size

                # Placeholder for cell type detection (needs custom classifier)
                # For now, assume all are generic "cell"
                cell_types = {"generic_cell": num_cells}

                visualize_segmentation(img, masks)

                # Store the metadata
                metadata.append({
                    "patch_id": file,
                    "num_cells": num_cells,
                    "mean_cell_area": mean_cell_area,
                    "cell_density": cell_density,
                    "cell_types": cell_types
                })

    #metadata_df = pd.DataFrame(metadata)
    #metadata_df.to_csv(output_csv, index=False)

    #print(f"Metadata saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", 
                        type=str,
                        default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam/images/tumor_048_tumor/0/"
    )
    parser.add_argument("--output_csv", 
                        type=str
    )

    args = parser.parse_args()
    process_patches(args.input_dir, args.output_csv)

