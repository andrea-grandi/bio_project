import os
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import seaborn as sns
from cellpose import models, io, plot
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import pdist, squareform


def segment_cells(image_path, channels=None, diameter=None, model_type='cyto2'):
    """
    Execute cellpose segmentation
    
    Params:
        - image_path: image path
        - channels:
        - diameter: threshold diameter 
        - model_type: model type for cellpose
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
        # Model initialization
        model = models.Cellpose(gpu=False, model_type=model_type)
        
        # Debug: image info
        print(f"Immagine shape: {img.shape}")
        print(f"Immagine dtype: {img.dtype}")
        
        try:
            masks, flows, styles, diams = model.eval([img], diameter=diameter, channels=channels)
            return masks, flows, styles, diams
        
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
    plt.imshow(np.squeeze(masks[0]), alpha=0.7, cmap='viridis')
    plt.title('Segmented Image')
    
    plt.tight_layout()
    plt.show()

def analyze_segmentation(img, masks):
    """
    Analyze Segmentation Data
    
    Params:
        - img: original image
        - masks: segmentation masks
    """
    if masks is None or len(masks) == 0:
        print("Masks NOT VALID")
        return

    # Number of Segmented Cells
    num_cells = (len(np.unique(masks[0])) - 1)  # -1 because of the background
    
    print(f"Number of Segmented Cells: {num_cells}")
    
    # Cells Sizes
    cell_sizes = [np.sum(masks[0] == i) for i in range(1, num_cells + 1)]
    
    print("Cells Size Distributions:")
    print(f"- Average Dimension: {np.mean(cell_sizes):.2f}")
    print(f"- Min Dimension: {np.min(cell_sizes)}")
    print(f"- Max Dimension: {np.max(cell_sizes)}")
    
    # Centroids
    centroids = [center_of_mass(masks[0]==i) for i in range(1, num_cells + 1)] # Centroid is a List of np arrays
    #print(f"Centroidi: {centroids}")
    
    # Distances
    distances = squareform(pdist(centroids))
    print(f"Cells distances:\n{distances}")
    
    return centroids
    
def plot_centroids(img, centroids):
    """
    Visualize Centroids on the Image
    
    Params:
        - img: input image array
        - centroids: list of tuples (x, y) representing centroids
    """
    plt.figure(figsize=(10, 8))
    
    # Original Image
    plt.imshow(img, cmap='gray')
    
    # Extract X and Y coordinates from centroids
    centroids_array = np.array(centroids)
    x_coords = centroids_array[:, 1]
    y_coords = centroids_array[:, 0] 
    
    plt.scatter(x_coords, y_coords, color='red', marker='o', label='Centroidi')
    
    plt.title('Centroidi delle cellule')
    plt.xlabel('Asse X')
    plt.ylabel('Asse Y')
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    image_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/data/camelyon17_v1.0/patches/patient_088_node_1/patch_patient_088_node_1_x_3232_y_16768.png"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    masks, flows, styles, diams = segment_cells(image_path, diameter=8)
    visualize_segmentation(img, masks)
    centroids = analyze_segmentation(img, masks)
    plot_centroids(img, centroids)
