import os
import subprocess

def sort_hierarchy(source_dir, dest_dir):
  """  
  Sort hierarchy 
  """
  os.makedirs(dest_dir, exist_ok=True)

  script_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/preprocessing/sort_hierarchy.py"

  command = [
      "python", script_path,
      "--sourcex20", source_dir,
      "--dest", dest_dir
  ]

  try:
      subprocess.run(command, check=True)
      print(f"Sort complete. Results saved to: {dest_dir}")
  except subprocess.CalledProcessError as e:
      print(f"Error during sorting process: {e}")
  except Exception as e:
      print(f"Unexpected error: {e}")

