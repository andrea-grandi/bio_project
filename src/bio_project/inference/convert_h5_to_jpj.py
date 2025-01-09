import os
import subprocess

def convert_h5_to_jpg(output_dir: str="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/input_patches", 
                      source_dir: str="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam/patches", 
                      slide_ext: str=".svs"):
  """  
  Convert h5 files to jpg
  """
  os.makedirs(output_dir, exist_ok=True)

  script_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/preprocessing/convert_h5_to_jpg.py"
  csv_file = "/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/output_clam"
  
  command = [
      "python", script_path,
      "--output_dir", output_dir,
      "--source_dir", source_dir,
      "--slide_ext", slide_ext,
      "--csv_file", csv_file
  ]

  try:
      subprocess.run(command, check=True)
      print(f"Conversion from h5 to jpg completed. Results saved to: {output_dir}")
  except subprocess.CalledProcessError as e:
      print(f"Error during conversion from h5 to jpg: {e}")
  except Exception as e:
      print(f"Unexpected error: {e}")
      