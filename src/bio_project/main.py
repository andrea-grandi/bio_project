import pandas as pd
import argparse
import os
import tensorflow as tf
import numpy as np
import gc
import random

from inference.preprocess_wsi import preprocess_wsi
from inference.convert_h5_to_jpg import convert_h5_to_jpg
from inference.sort_hierarchy import sort_hierarchy
from inference.feature_extraction import feature_extraction
from inference.create_csv import create_csv
from inference.prepare_dataset import prepare_dataset
from inference.inference import inference

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
   arg.add_argument("--source_dir", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/input_slide", help="Path to WSI directory")
   arg.add_argument("--save_dir", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam", help="Path to CLAM save directory")
   arg.add_argument("--preset", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/preprocessing/CLAM/presets/bwh_biopsy.csv", help="Path to CLAM preset")
   arg.add_argument("--patch_size", type=int, default=256, help="Patch size")

   # Args for converting h5 to jpg
   arg.add_argument("--output_dir", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam", help="Where to save the patches")
   arg.add_argument("--csv_file", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam/process_list_autogen.csv", help="Path to the csv file")
   arg.add_argument("--slide_ext", type=str, default=".tif", help="Slide extension (e.g. .svs, .tif)")

   # Args for sorting hierarchy
   arg.add_argument("--sourcex20", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam/images", help="Source folder")
   arg.add_argument("--dest", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam_sorted", help="Destination folder")

   # Args for feature extraction
   arg.add_argument("--extractedpatchespath", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam_sorted", help="Path to extracted patches")
   arg.add_argument("--savepath", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_feats", help="Path to save extracted features")
   arg.add_argument("--pretrained_weights1", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/weights/dino_camelyon16/checkpoint_20x.pth", help="Path to pretrained weights 20x")
   arg.add_argument("--pretrained_weights2", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/weights/dino_camelyon16/checkpoint_10x.pth", help="Path to pretrained weights 10x")
   arg.add_argument("--pretrained_weights3", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/weights/dino_camelyon16/checkpoint_5x.pth", help="Path to pretrained weights 5x")
   arg.add_argument("--propertiescsv", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/cam_multi.csv", help="Path to properties csv")

   # Args for preparing dataset
   arg.add_argument("--source_feats", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_feats", help="Features folder")
   arg.add_argument("--dest_feats", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/final_embeddings", help="Final embeddings folder")
   #arg.add_argument("--levels", type=int, default=[3], help="Patch levels")

   # Args for inference


   args = arg.parse_args()


   ### ----- SET SEED ----- ###
   set_seed()

   ### ----- PREPROCESSING WSI ----- ###
   preprocess_wsi(args.source_dir, args.save_dir, args.patch_size, args.preset)

   ### ----- CONVERT H5 TO JPG ----- ###
   convert_h5_to_jpg(args.output_dir, args.source_dir, args.slide_ext, args.csv_file)

   ### ----- SORT HIERARCHY ----- ###
   sort_hierarchy(args.sourcex20, args.dest)

   ### ----- CREATE CSV ----- ###
   create_csv()

   ### ----- FEATURE EXTRACTION ----- ###
   feature_extraction(args.extractedpatchespath, args.savepath, args.pretrained_weights1, args.pretrained_weights2, args.pretrained_weights3, args.propertiescsv)

   ### ----- PREPARE DATASET ----- ###
   prepare_dataset(args.source_feats, args.dest_feats)

   ### ----- INFERENCE ----- ###
   inference()

if __name__ == "__main__":
   main()