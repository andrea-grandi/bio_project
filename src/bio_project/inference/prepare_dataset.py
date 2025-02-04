import os
import subprocess

def prepare_dataset(source_feats, dest_feats):
  """  
  Prepare dataset for training and testing
  """
  os.makedirs(dest_feats, exist_ok=True)

  script_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/feature_extraction/prepare_dataset.py"

  command = [
      "python", script_path,
      "--source", source_feats,
      "--dest", dest_feats
  ]

  try:
      subprocess.run(command, check=True)
      print(f"Dataset preparation complete. Results saved to: {dest_feats}")
  except subprocess.CalledProcessError as e:
      print(f"Error during dataset preparation: {e}")
  except Exception as e:
      print(f"Unexpected error: {e}")
