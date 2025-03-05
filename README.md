<div align="center">

![Release](https://img.shields.io/github/v/tag/andrea-grandi/bio_project.svg?sort=semver)
![Latest commit](https://img.shields.io/github/last-commit/andrea-grandi/bio_project)

# **Artificial Intelligence in Bioinformatics Project**

</div>

## Overview
The visual examination of histopathological images is a cornerstone of cancer diagnosis, requiring pathologists to analyze tissue sections 
across multiple magnifications to identify tumor cells and subtypes. However, existing attention-based Multiple Instance Learning (MIL) models 
for Whole Slide Image (WSI) analysis often neglect contextual and numerical features, resulting in limited interpretability and potential misclassifications. 
Furthermore, the original MIL formulation incorrectly assumes the patches of the same image to be independent, leading to a loss of spatial 
context as information flows through the network. Incorporating contextual knowledge into predictions is particularly important given the 
inclination for cancerous cells to form clusters and the presence of spatial indicators for tumors. To address these limitations, we propose an enhanced
MIL framework that integrates pre-contextual numerical information derived from semantic segmentation. Specifically, our approach combines visual
features with nuclei-level numerical attributes, such as cell density and morphological diversity, extracted using advanced segmentation tools like Cellpose.
These enriched features are then fed into a modified BufferMIL model for WSI classification. We evaluate our method on detecting lymph node metastases 
(CAMELYON16 and TCGA lung).

## Folder Structure
- `src/`: contains the source code for the project
- `presentation/`: contains the presentation slides and images
- `notebooks/`: contains the Jupyter notebooks for this project
- `reports/`: contains the project paper
- `references/`: contains the references

## Prerequisites
Before running, ensure that:
- The project dependencies are installed (`requirements.txt` or `environment.yml`)
- You have the necessary pretrained weights for feature extraction
- Whole Slide Image (WSI) is placed in the correct input directory (PATH_TO_INPUT_SLIDE)

## Installation
To install the required dependencies, run the following command:
```bash
conda create -n bio python=3.10
conda activate bio
conda env update --file environment.yml
```

### Dataset
Camelyon16/TCGA

### Data Preprocessing
This work uses [CLAM](https://github.com/mahmoodlab/CLAM) to filter out background patches. 
After the .h5 coordinate generation, use:

- [H5 to JPG](src/bio_project/preprocessing/convert_h5_to_jpg.py): to convert the .h5 files to .jpg
- [Sort Images](src/bio_project/preprocessing/sort_hierarchy.py): to sort the images in the correct folders
- [Dino Training](https://github.com/facebookresearch/dino): given the patches, train dino with the `vit_small` option
- [Feature Extraction](src/bio_project/feature_extraction.py): it extracts patch features and adjacency matrices

### Model
- Architectures: [CellPose](https://github.com/MouseLand/cellpose) | [DINO](https://github.com/facebookresearch/dino) | [BufferMIL](https://github.com/aimagelab/mil4wsi)
- Pretrained Weights: [Pretrained Checkpoints Used](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/camelyon16/dino/x20/checkpoint.pth.gz)
- Input size: `[256 x 256]`

### Proposed Architectures
![First](presentation/images/custom_arch_V1.png)
![Second](presentation/images/custom_arch_V2.png)

### Hyperparameters
| Parameter      | Value |
|--------------|-------|
| Learning Rate | 0.001 |
| Optimizer    | Adam |
| Loss Function | BCEWithLogitsLoss |
| Epochs       | 200 |
| Buffer size  | 10 |
| Buffer rate  | 10 |
| Buffer aggregate | Mean |

### Training and Inference

For training and inference look at `src/bio_project` and `src/bio_project/inference` folders.

## Results
Include visual results from different stages:
- Sample WSI patches before/after preprocessing
- Feature extraction outputs
- Training loss and accuracy curves

### 1. Sample Extracted Patches
| Original WSI | Extracted Patches |
|-------------|-----------------|
| ![WSI Example](src/bio_project/inference/output_clam/masks/slide_404.jpg) | ![Patches Example](src/bio_project/inference/output_clam/images/tumor_048_tumor/0/_x_18240_y_192000.jpg) |

### 2. CellPose
![Cellpose](presentation/images/cellpose_example_3.png)

### 3. Training Loss and Accuracy Curves
![Training Curves](presentation/images/loss.png)
![Example](presentation/images/comparison_between_all.png)


## Credits

- Andrea Grandi: [@andrea-grandi](https://github.com/andrea-grandi)
- Daniele Vellani: [@franzione1](https://github.com/franzione1)

