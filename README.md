<div align="center">

![Release](https://img.shields.io/github/v/tag/andrea-grandi/bio_project.svg?sort=semver)
![Latest commit](https://img.shields.io/github/last-commit/andrea-grandi/bio_project)

# **Artificial Intelligence in Bioinformatics Project**

</div>

## Overview
The visual examination of histopathological images is a cornerstone of cancer diagnosis, requiring pathologists to analyze tissue sections across multiple magnifications to identify tumor cells and subtypes. However, existing attention-based Multiple Instance Learning (MIL) models for Whole Slide Image (WSI) analysis often neglect contextual and numerical features, resulting in limited interpretability and potential misclassifications. Furthermore, the original MIL formulation incorrectly assumes the patches of the same image to be independent, leading to a loss of spatial context as information flows through the network. Incorporating contextual knowledge into predictions is particularly important given the inclination for cancerous cells to form clusters and the presence of spatial indicators for tumors. To address these limitations, we propose an enhanced MIL framework that integrates pre-contextual numerical information derived from semantic segmentation. Specifically, our approach combines visual features with nuclei-level numerical attributes, such as cell density and morphological diversity, extracted using advanced segmentation tools like Cellpose. These enriched features are then fed into a modified BufferMIL model for WSI classification. We evaluate our method on subtyping non-small cell lung cancer (TCGA-NSCLC) and detecting lymph node metastases (CAMELYON16 and CAMELYON17).

## Prerequisites
Before running inference, ensure that:
- The project dependencies are installed (`requirements.txt` or `environment.yml`)
- You have the necessary pretrained weights for feature extraction
- Whole Slide Images (WSI) are placed in the correct input directory

## Training Process

### Dataset
- Name: [Camelyon16]
- Number of slides: [300]

### Model
- Architecture: [DINO / Transformer / MIL-based Model]
- Pretrained Weights: [Pretrained Checkpoints Used]
- Input size: `[256 x 256]`

### Hyperparameters
| Parameter      | Value |
|--------------|-------|
| Learning Rate | [0.001] |
| Optimizer    | [Adam / SGD] |
| Loss Function | [BCEWithLogitsLoss] |
| Epochs       | [200] |

### 7. Figures
Include visual results from different stages:
- Sample WSI patches before/after preprocessing
- Feature extraction outputs (e.g., t-SNE visualization)
- Training loss and accuracy curves

## Figures

### 1. Sample Extracted Patches
| Original WSI | Extracted Patches |
|-------------|-----------------|
| ![WSI Example](path_to_example_wsi) | ![Patches Example](path_to_example_patches) |

### 2. Feature Extraction t-SNE Visualization
![t-SNE Visualization](path_to_tsne_plot)

### 3. Training Loss and Accuracy Curves
![Training Curves](path_to_training_curves)

## Inference

### 1. Run the Inference Script
To start the inference process, execute the following command:
```bash
python src/bio_project/main.py \
  --source_dir PATH_TO_INPUT_SLIDE \
  --save_dir PATH_TO_OUTPUT_CLAM \
  --preset PATH_TO_PRESETS \
  --patch_size PATCH_SIZE
```
This extracts patches from WSIs using the CLAM preset.

### 2. Convert Extracted Patches to JPG
```bash
python src/bio_project/main.py \
  --output_dir PATH_TO_OUTPUT_CLAM \
  --csv_file PATH_TO_CSV_FILE \
  --slide_ext SLIDE_EXTENSION
```
This step converts HDF5 patches into JPEG images for easier visualization.

### 3. Sort Image Hierarchy
```bash
python src/bio_project/main.py \
  --sourcex20 PATH_TO_SOURCE_IMAGES \
  --dest PATH_TO_SORTED_OUTPUT
```
This organizes the extracted patches into a structured hierarchy.

### 4. Extract Features Using DINO
```bash
python src/bio_project/main.py \
  --extractedpatchespath PATH_TO_EXTRACTED_PATCHES \
  --savepath PATH_TO_FEATURES_OUTPUT \
  --pretrained_weights1 PATH_TO_PRETRAINED_WEIGHTS_20X \
  --pretrained_weights2 PATH_TO_PRETRAINED_WEIGHTS_10X \
  --pretrained_weights3 PATH_TO_PRETRAINED_WEIGHTS_5X \
  --propertiescsv PATH_TO_PROPERTIES_CSV
```
This extracts visual features from the patches using pretrained DINO models.

### 5. Prepare Final Dataset
```bash
python src/bio_project/main.py \
  --source_feats PATH_TO_FEATURES_FOLDER \
  --dest_feats PATH_TO_FINAL_EMBEDDINGS
```
This step aggregates the extracted features into the final dataset for model inference.

## Credits

- Andrea Grandi: [@andrea-grandi](https://github.com/andrea-grandi)
- Daniele Vellani: [@franzione1](https://github.com/franzione1)

