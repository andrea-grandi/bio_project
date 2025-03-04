<div align="center">

![Release](https://img.shields.io/github/v/tag/andrea-grandi/bio_project.svg?sort=semver)
![Latest commit](https://img.shields.io/github/last-commit/andrea-grandi/bio_project)

# **Artificial Intelligence in Bioinformatics Project**

</div>

## Credits

- Andrea Grandi: [@andrea-grandi](https://github.com/andrea-grandi)
- Daniele Vellani: [@franzione1](https://github.com/franzione1)

The visual examination of histopathological images is a cornerstone of cancer diagnosis, requiring pathologists to analyze tissue sections across multiple magnifications to identify tumor cells and subtypes. However, existing attention-based Multiple Instance Learning (MIL) models for Whole Slide Image (WSI) analysis often neglect contextual and numerical features, resulting in limited interpretability and potential misclassifications. Furthermore, the original MIL formulation incorrectly assumes the patches of the same image to be independent, leading to a loss of spatial context as information flows through the network. Incorporating contextual knowledge into predictions is particularly important given the inclination for cancerous cells to form clusters and the presence of spatial indicators for tumors. To address these limitations, we propose an enhanced MIL framework that integrates pre-contextual numerical information derived from semantic segmentation. Specifically, our approach combines visual features with nuclei-level numerical attributes, such as cell density and morphological diversity, extracted using advanced segmentation tools like Cellpose. These enriched features are then fed into a modified BufferMIL model for WSI classification. We evaluate our method on subtyping non-small cell lung cancer (TCGA-NSCLC) and detecting lymph node metastases (CAMELYON16 and CAMELYON17).


# Inference Guide for Bio_Project

## Overview
This guide provides step-by-step instructions for performing inference using the `main.py` script in the Bio_Project repository.

## Prerequisites
Before running inference, ensure that:
- The project dependencies are installed (`requirements.txt` or `environment.yml`)
- You have the necessary pretrained weights for feature extraction
- Whole Slide Images (WSI) are placed in the correct input directory

## Steps for Inference

### 1. Run the Inference Script
To start the inference process, execute the following command:
```bash
python src/bio_project/main.py \
  --source_dir /Users/andreagrandi/Developer/bio_project/src/bio_project/inference/input_slide \
  --save_dir /Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam \
  --preset /Users/andreagrandi/Developer/bio_project/src/bio_project/preprocessing/CLAM/presets/bwh_biopsy.csv \
  --patch_size 256
```
This extracts patches from WSIs using the CLAM preset.

### 2. Convert Extracted Patches to JPG
```bash
python src/bio_project/main.py \
  --output_dir /Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam \
  --csv_file /Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam/process_list_autogen.csv \
  --slide_ext .tif
```
This step converts HDF5 patches into JPEG images for easier visualization.

### 3. Sort Image Hierarchy
```bash
python src/bio_project/main.py \
  --sourcex20 /Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam/images \
  --dest /Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam_sorted
```
This organizes the extracted patches into a structured hierarchy.

### 4. Extract Features Using DINO
```bash
python src/bio_project/main.py \
  --extractedpatchespath /Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam_sorted \
  --savepath /Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_feats \
  --pretrained_weights1 /Users/andreagrandi/Developer/bio_project/src/bio_project/weights/dino_camelyon16/checkpoint_20x.pth \
  --pretrained_weights2 /Users/andreagrandi/Developer/bio_project/src/bio_project/weights/dino_camelyon16/checkpoint_10x.pth \
  --pretrained_weights3 /Users/andreagrandi/Developer/bio_project/src/bio_project/weights/dino_camelyon16/checkpoint_5x.pth \
  --propertiescsv /Users/andreagrandi/Developer/bio_project/src/bio_project/inference/cam_multi.csv
```
This extracts visual features from the patches using pretrained DINO models.

### 5. Prepare Final Dataset
```bash
python src/bio_project/main.py \
  --source_feats /Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_feats \
  --dest_feats /Users/andreagrandi/Developer/bio_project/src/bio_project/inference/final_embeddings
```
This step aggregates the extracted features into the final dataset for model inference.

## Adding Results to README.md
To include inference results in `README.md`, add the following section:

```markdown
## Inference Results

### Sample Extracted Patches
| Original WSI | Extracted Patches |
|-------------|-----------------|
| ![WSI Example](path_to_example_wsi) | ![Patches Example](path_to_example_patches) |

### Feature Extraction Output
Extracted feature dimensions: `[Batch_Size, Feature_Size]`

Example JSON Output:
```json
{
  "slide_id": "example.tif",
  "num_patches": 500,
  "features_shape": [500, 1024]
}
```
```

Replace `path_to_example_wsi` and `path_to_example_patches` with actual paths to the images.


