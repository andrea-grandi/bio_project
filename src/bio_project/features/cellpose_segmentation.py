import numpy as np 
import matplotlib.pyplot as plt 
from cellpose import models, io, utils 


def segment_cells(image_path, diameter=None, model_type='cyto'):
    """
    Execute cellpose segmentation

    Parameters:
    - image_path: image system path of image to segment 
    - diameter: mean diameter of cell 
    - model_type: model type of cell ('cyto' for cytoplasmatic cells, 'nuclei' for nuclear cells) 

    Returns: 
    - masks: segmantation masks
    - flows: segmentation flows 
    - styles: segmantation styles 
    """

    img = io.imread(image_path) # image read cellpose method
    model = model.cellpose(model_type=model_type)

    masks, flows, styles = model.eval([img],
                                      diameter=diameter,
                                      flow_threshold=0.4,
                                      cellpose_threshold=0.0)

    return mask, flows, styles

def visualize_segmentation(img, masks):
    plt.figure(figsize=(12,5))
    plt.subplot((121))
    plt.imshow(img)
    plt.title('Original Image')
    
    plt.subplot(122)
    plt.imshow(img, alpha=0.5)
    plt.imshow(masks[0], alpha=0.5, cmap='viridis')
    plt.title('Segmented Image')
    
    plt.tight_layout()
    plt.show()

def analyze_segmentation(masks):
    """
    Analyze Segmentation Data
    """
    # Number of Segmented Cells
    num_cells = len(np.unique(masks[0])) - 1  # -1 because of the background
    
    print(f"Number of Segmented Cells: {num_cells}")
    
    cell_sizes = [np.sum(masks[0] == i) for i in range(1, num_cells + 1)]
    
    print("Distribuzione delle dimensioni delle cellule:")
    print(f"- Dimensione media: {np.mean(cell_sizes):.2f}")
    print(f"- Dimensione minima: {np.min(cell_sizes)}")
    print(f"- Dimensione massima: {np.max(cell_sizes)}")

if __name__ == "__main__":
    image_path = "../data/test_wsi/whole_slide_image.svs" 
    masks, flows, styles = segment_cells(image_path, diameter=50)
    visualize_segmentation(io.imread(image_path), masks)
    analyze_segmentation(masks)

