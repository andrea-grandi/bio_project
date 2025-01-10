import pandas as pd
import argparse
import os
import tensorflow as tf
import numpy as np
import gc
import random

from inference.preprocess_wsi import preprocess_wsi
from inference.convert_h5_to_jpg import convert_h5_to_jpg
from inference.cellpose_feature_extraction import cellpose_feature_extraction
from inference.feature_extraction import feature_extraction

def set_seed(seed: int = 42) -> None:
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DISABLE_SPARSE_SOFTMAX_XENT_WITH_LOGITS_OP_DETERMINISM_EXCEPTIONS']='1'
  

def main():
   ### ----- PARSING ARGUMENTS ----- ###
   arg = argparse.ArgumentParser()
   
   # Args for CLAM preprocessing
   arg.add_argument("--source_dir", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/input_wsi", help="Path to WSI directory")
   arg.add_argument("--save_dir", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam", help="Path to CLAM save directory")
   arg.add_argument("--patch_size", type=int, default=256, help="Patch size")
   
   # Args for converting h5 to jpg
   arg.add_argument("--output_dir", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam", help="Where to save the patches")
   arg.add_argument("--slide_ext", type=str, default=".svs", help="Slide extension (e.g. .svs, .tif)")

   # Args for cellpose feature extraction
   arg.add_argument("--input_cellpose_dir", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam/images", help="Where to save the metadata")
   arg.add_argument("--output_cellpose_csv", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/metadata/metadata.csv", help="Where to save the metadata")

   # Args for feature extraction
   #arg.add_argument("--numerical_features_path", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/metadata/cellpose_metadata.csv", help="Path to the metadata")
   #arg.add_argument("--output_pt_path", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/features/concatenated_features.pt", help="Where to save the features")
   
   args = arg.parse_args()

   ### ----- SET SEED ----- ###
   set_seed()

   ### ----- PREPROCESSING WSI ----- ###
   preprocess_wsi(args.source_dir, args.save_dir, args.patch_size)

   ### ----- CONVERT H5 TO JPG ----- ###
   convert_h5_to_jpg(args.output_dir, args.source_dir, args.slide_ext)

   ### ----- CELLPOSE FEATURE EXTRACTION ----- ###
   cellpose_feature_extraction(args.input_cellpose_dir, args.output_cellpose_csv)

   ### ----- FEATURE EXTRACTION ----- ###
   #feature_extraction(args.output_dir, args.numerical_features_path, args.output_pt_path)


if __name__ == "__main__":
   main()