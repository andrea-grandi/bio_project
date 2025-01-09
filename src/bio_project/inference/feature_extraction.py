import os
import subprocess

def feature_extraction(image_dir: str="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/input_patches", 
                       numerical_features_path: str="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/metadata/cellpose_metadata.csv", 
                       output_pt_path: str="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/features/concatenated_features.pt"):
  """
  Feature extraction using DINO
  """
  os.makedirs(output_pt_path, exist_ok=True)

  script_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/feature_extraction/feature_extractor_dino.py"
  
  command = [
      "python", script_path,
      "--image_dir", image_dir,
      "--numerical_features_path", numerical_features_path,
      "--output_pt_path", output_pt_path
  ]

  try:
      subprocess.run(command, check=True)
      print(f"Feature extraction completed. Results saved to: {output_pt_path}")
  except subprocess.CalledProcessError as e:
      print(f"Error during feature extraction: {e}")
  except Exception as e:
      print(f"Unexpected error: {e}")
      
