import numpy as np

from scipy.spatial.distance import squareform, pdist
from scipy.ndimage import center_of_mass


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
    
    # Calculate Cell Density (cells per pixel area)
    total_area = img.shape[0] * img.shape[1]  # Total image area in pixels
    cell_density = num_cells / total_area
    print(f"Cell Density: {cell_density:.6f} cells/pixel²")
    
    # Optional: Cell Density in cells/mm² (if you know the pixel size)
    # Uncomment and adjust the pixel_size value according to your microscope settings
    # pixel_size = 1.0  # size of one pixel in micrometers
    # area_mm2 = (total_area * (pixel_size/1000)**2)  # Convert to mm²
    # cell_density_mm2 = num_cells / area_mm2
    # print(f"Cell Density: {cell_density_mm2:.2f} cells/mm²")
    
    # Cells Sizes
    cell_sizes = [np.sum(masks[0] == i) for i in range(1, num_cells + 1)]

    print("Cells Size Distributions:")
    print(f"- Average Dimension: {np.mean(cell_sizes):.2f}")
    print(f"- Min Dimension: {np.min(cell_sizes)}")
    print(f"- Max Dimension: {np.max(cell_sizes)}")
