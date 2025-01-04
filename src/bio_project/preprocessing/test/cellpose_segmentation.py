import cv2
import argparse

from cellpose import models, io

from utils.visualize import plot_centroids, visualize_segmentation
from utils.analyze import analyze_segmentation


def segment_cells(img, channels=None, diameter=None, model_type='cyto3'):
    """
    Execute cellpose segmentation
    
    Params:
        - img: input image
        - channels: channels for Cellpose
        - diameter: threshold diameter, auto-calculated if None
        - model_type: model type for Cellpose
    """
    try:
        # Model initialization
        model = models.Cellpose(gpu=False, model_type=model_type)
        
        # Debug: image info
        print(f"Immagine shape: {img.shape}")
        print(f"Immagine dtype: {img.dtype}")
        
        # Auto-estimate diameter if not provided
        if diameter is None:
            estimated_diameter = model.eval(img, channels=channels, diameter=None)[-1]
            print(f"Diametro stimato: {estimated_diameter}")
            diameter = estimated_diameter
        
        try:
            masks, flows, styles, diams = model.eval(img, 
                                                     diameter=diameter, 
                                                     channels=channels)
            return masks, flows, styles, diams
        
        except Exception as e:
            print(f"Segmentation Error: {e}")
            return None, None, None, None
        
    except Exception as e:
        print(f"Image Loading Error: {e}")
        return None, None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", 
                        type=str, 
                        default="/Users/andreagrandi/Developer/bio_project/src/bio_project/dataset/camelyon17/512x512_patches/patch_patient_015_node_2_x_2304_y_27776.png"
    )

    args = parser.parse_args()
    image_path = args.image_path
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Dynamic diameter estimation
    masks, flows, styles, diams = segment_cells(img)

    visualize_segmentation(img, masks)
    analyze_segmentation(img, masks)
    plot_centroids(img, masks)