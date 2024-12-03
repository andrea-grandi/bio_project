import os
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
from cellpose import models, io

def segment_cells(image_path, diameter=None, model_type='cyto'):
    """
    Execute cellpose segmentation
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
        # Model initialization
        model = models.Cellpose(model_type=model_type)
        
        # Debug: image info
        print(f"Immagine shape: {img.shape}")
        print(f"Immagine dtype: {img.dtype}")
        
        try:
            masks = model.eval([img], 
                                diameter=diameter, 
                                flow_threshold=0.9,
                                cellprob_threshold=0.4)

            return masks
        
        except Exception as e:
            print(f"Segmentation Error: {e}")
            return None, None, None
    
    except Exception as e:
        print(f"Image Loading Error: {e}")
        return None, None, None

def visualize_segmentation(img, masks):
    if img is None or masks is None or len(masks) == 0:
        print("Impossible to Visualize Image - NOT VALID")
        return
    
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(122)
    plt.imshow(img, alpha=0.5, cmap='gray')
    plt.imshow(np.squeeze(masks[0]), alpha=0.5, cmap='viridis')
    plt.title('Segmented Image')
    
    plt.tight_layout()
    plt.show()

def analyze_segmentation(masks):
    """
    Analyze Segmentation Data
    """
    if masks is None or len(masks) == 0:
        print("Masks NOT VALID")
        return
    
    # Number of Segmented Cells
    num_cells = len(np.unique(masks[0])) - 1  # -1 because of the background
    
    print(f"Number of Segmented Cells: {num_cells}")
    
    cell_sizes = [np.sum(masks[0] == i) for i in range(1, num_cells + 1)]
    
    print("Cells Size Distributions:")
    print(f"- Average Dimension: {np.mean(cell_sizes):.2f}")
    print(f"- Min Dimension: {np.min(cell_sizes)}")
    print(f"- Max Dimension: {np.max(cell_sizes)}")

if __name__ == "__main__":
    image_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/data/camelyon17_v1.0/patches/patient_020_node_2/patch_patient_020_node_2_x_1344_y_36928.png"
    
    masks = segment_cells(image_path, diameter=60)
    
    if masks is not None:
        visualize_segmentation(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), masks)
        analyze_segmentation(masks)
    else:
        print("The masks are empty")
