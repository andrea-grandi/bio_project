import os
import subprocess

def feature_extraction(extractedpatchespath, savepath, pretrained_weights1, pretrained_weights2, pretrained_weights3, propertiescsv):
  """
  Feature extraction using DINO and Cellpose.
  """
  os.makedirs(savepath, exist_ok=True)

  script_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/feature_extraction/run_with_submitit.py"
  
  command = [
        "python", script_path,
        "--extractedpatchespath", extractedpatchespath,
        "--savepath", savepath,
        "--pretrained_weights1", pretrained_weights1,
        "--pretrained_weights2", pretrained_weights2,
        "--pretrained_weights3", pretrained_weights3,
        "--propertiescsv", propertiescsv
  ]

  try:
      subprocess.run(command, check=True)
      print(f"Feature extraction completed. Results saved to: {savepath}")
  except subprocess.CalledProcessError as e:
      print(f"Error during feature extraction: {e}")
  except Exception as e:
      print(f"Unexpected error: {e}")
      
