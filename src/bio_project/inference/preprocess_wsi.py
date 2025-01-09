import os
import subprocess

def preprocess_wsi(source_dir: str="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/input_wsi", 
                   save_dir: str="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam",
                   patch_size: int = 256):
  """  
  Preprocess WSI with CLAM 
  """
  os.makedirs(save_dir, exist_ok=True)

  script_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/preprocessing/CLAM/create_patches_fp.py"
  preset = "/Users/andreagrandi/Developer/bio_project/src/bio_project/preprocessing/CLAM/presets/bwh_biopsy.csv"

  command = [
      "python", script_path,
      "--source", source_dir,
      "--save_dir", save_dir,
      "--patch_size", str(patch_size),
      "--preset", preset,
      "--seg",
      "--patch",
      "--stitch"
  ]

  try:
      subprocess.run(command, check=True)
      print(f"CLAM preprocessing complete. Results saved to: {save_dir}")
  except subprocess.CalledProcessError as e:
      print(f"Error during CLAM preprocessing: {e}")
  except Exception as e:
      print(f"Unexpected error: {e}")

