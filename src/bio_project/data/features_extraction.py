"""
This file is for extract features from patch.

We use Detectron2 trained on a custom dataset for instance
segmentation and extract the masks and after compute nuclei
distribution, number of nuclei and cells classification.

With this we can pass the features to the bufferMIL
instead of using significant pathes like the original.

"""

import numpy as np
import torch
import cv2

import detectron2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from scipy.spatial import distance


class PatchFeatureExtractor:
    def __init__(self, model_path, config_path, num_classes=2):
        """
        Initialize feature extractor with pre-trained Detectron2 model
        
        Args:
            model_path (str): Path to the trained model weights
            config_path (str): Path to the model configuration file
            num_classes (int): Number of classes for segmentation
        """
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.DEVICE = 'cpu'

        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

    def segment_patch(self, image):
        """
        Perform instance segmentation on a patch
        
        Args:
            image (np.ndarray): Input image patch
        
        Returns:
            dict: Segmentation results with masks, classes, and scores
        """
        outputs = self.predictor(image)
        
        return {
            'masks': outputs['instances'].pred_masks.cpu().numpy(),
            'classes': outputs['instances'].pred_classes.cpu().numpy(),
            'scores': outputs['instances'].scores.cpu().numpy()
        }

    def compute_nuclei_distribution(self, masks):
        """
        Compute spatial distribution and statistics of nuclei
        
        Args:
            masks (np.ndarray): Binary masks of nuclei
        
        Returns:
            dict: Nuclei distribution features
        """
        # Compute centroid for each nucleus
        centroids = []
        for mask in masks:
            y, x = np.where(mask)
            centroids.append((np.mean(x), np.mean(y)))
        
        # Compute inter-nuclei distances
        distance_matrix = distance.cdist(centroids, centroids)
        
        return {
            'num_nuclei': len(masks),
            'centroids': centroids,
            'mean_distance': np.mean(distance_matrix[np.triu_indices(len(masks), k=1)]),
            'max_distance': np.max(distance_matrix),
            'min_distance': np.min(distance_matrix[distance_matrix > 0])
        }

    def classify_cells(self, masks, classes):
        """
        Classify cells based on segmentation results
        
        Args:
            masks (np.ndarray): Binary masks of cells/nuclei
            classes (np.ndarray): Predicted class indices
        
        Returns:
            dict: Cell classification statistics
        """
        class_counts = {}
        for cls in np.unique(classes):
            class_counts[cls] = np.sum(classes == cls)
        
        return {
            'class_distribution': class_counts,
            'dominant_class': max(class_counts, key=class_counts.get)
        }

    def extract_patch_features(self, patch):
        """
        Extract comprehensive features from a single patch
        
        Args:
            patch (np.ndarray): Input image patch
        
        Returns:
            dict: Comprehensive patch features
        """
        # Perform instance segmentation
        segmentation_results = self.segment_patch(patch)
        print("TEST")
        print(segmentation_results['masks'])
        
        # Compute nuclei distribution
        nuclei_distribution = self.compute_nuclei_distribution(segmentation_results['masks'])
        
        # Classify cells
        cell_classification = self.classify_cells(
            segmentation_results['masks'], 
            segmentation_results['classes']
        )
        
        return {
            'segmentation': segmentation_results,
            'nuclei_distribution': nuclei_distribution,
            'cell_classification': cell_classification
        }
    
    def visualize_segmentation(self, mask):
        """
        

        Parameters
        ----------
        mask : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """        
        

def main():
    # Example usage
    model_path = '/Users/andreagrandi/Developer/bio_project/models/model_10k_iter.pth'
    config_path = '/Users/andreagrandi/Developer/bio_project/models/config_10k_iter.yaml'
    
    extractor = PatchFeatureExtractor(model_path, config_path)
    
    # Load a sample patch
    sample_patch = cv2.imread('/Users/andreagrandi/Developer/bio_project/src/bio_project/data/camelyon17_v1.0/patches/patient_046_node_4/patch_patient_046_node_4_x_1728_y_17920.png')
    
    # Extract features
    patch_features = extractor.extract_patch_features(sample_patch)
    
    # You can now pass these features to bufferMIL
    print(patch_features)


if __name__ == '__main__':
    main()