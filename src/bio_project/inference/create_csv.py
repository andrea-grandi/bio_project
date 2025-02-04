import os
import subprocess

def create_csv():
  """  
  Create CSV
  """
  script_path = "/Users/andreagrandi/Developer/bio_project/src/bio_project/feature_extraction/create_csv.py"

  command = [
      "python", script_path
  ]

  try:
      subprocess.run(command, check=True)
      print(f"CSV creation complete.")
  except subprocess.CalledProcessError as e:
      print(f"Error during CSV creation process: {e}")
  except Exception as e:
      print(f"Unexpected error: {e}")
