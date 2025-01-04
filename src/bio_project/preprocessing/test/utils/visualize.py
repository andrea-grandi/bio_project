import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import center_of_mass


def visualize_segmentation(img, masks):
    if img is None or masks is None or len(masks) == 0:
        print("Impossible to Visualize Image - NOT VALID")
        return

    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    
    print(masks.shape)
    
    plt.subplot(122)
    plt.imshow(img, alpha=0.5, cmap='gray')
    plt.imshow(masks, alpha=0.7, cmap='viridis')
    plt.title('Segmented Image')
    
    plt.tight_layout()
    plt.show()

def plot_centroids(img, masks):
    """
    Visualize Centroids on the Image
    
    Params:
        - img: input image array
        - masks: list of masks
    """
    if img is None or masks is None or len(masks) == 0:
        print("Impossible to Visualize Image - NOT VALID")
        return
    
    num_cells = (len(np.unique(masks)) - 1)
    centroids = [center_of_mass(masks==i) for i in range(1, num_cells + 1)]

    plt.figure(figsize=(10, 8))
    
    # Original Image
    plt.imshow(img, cmap='gray')
    
    # Extract X and Y coordinates from centroids
    centroids_array = np.array(centroids)
    x_coords = centroids_array[:, 1]
    y_coords = centroids_array[:, 0] 
    
    plt.scatter(x_coords, y_coords, color='red', marker='o', label='Centroids')
    
    plt.title('Cells Centroids')
    plt.legend()
    plt.show()