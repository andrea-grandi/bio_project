import os
import subprocess

def cellpose_feature_extraction(input_dir: str="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/input_patches", 
                                output_csv: str="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/metadata"):
  """
  Cellpose numerical feature extraction
  """
  os.makedirs(output_csv, exist_ok=True)

  script_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/preprocessing/cellpose_segmentation.py"
  
  command = [
    "python", script_path,
    "--input_dir", input_dir,
    "--output_csv", output_csv
  ]

  try:
      subprocess.run(command, check=True)
      print(f"CellPose segmentation completed. Results saved to: {output_csv}")
  except subprocess.CalledProcessError as e:
      print(f"Error during cellpose segmentation: {e}")
  except Exception as e:
      print(f"Unexpected error: {e}")
