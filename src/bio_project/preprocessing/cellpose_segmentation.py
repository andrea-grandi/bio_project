import cv2
import argparse
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 

from cellpose import models

from utils.visualize import plot_centroids, visualize_segmentation
from utils.analyze import analyze_segmentation


def segment_cells(img, channels=None, diameter=None, model_type='cyto2'):
    """
    Execute cellpose segmentation
    
    Params:
        - img: input image
        - channels:
        - diameter: threshold diameter 
        - model_type: model type for cellpose
    """
    try:
        # Model initialization
        model = models.Cellpose(gpu=False, model_type=model_type)
        
        # Debug: image info
        print(f"Immagine shape: {img.shape}")
        print(f"Immagine dtype: {img.dtype}")
        
        try:
            masks, flows, styles, diams = model.eval(img, 
                                                     diameter=diameter, 
                                                     channels=channels
                                                     )
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
                        default="/Users/andreagrandi/Developer/bio_project/src/bio_project/tests/dataset/patches/patch_patient_004_node_4_x_10112_y_18816.png"
                        )

    args = parser.parse_args()
    image_path = args.image_path
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    masks, flows, styles, diams = segment_cells(img, diameter=8)

    visualize_segmentation(img, masks)
    analyze_segmentation(img, masks)
    plot_centroids(img, masks)