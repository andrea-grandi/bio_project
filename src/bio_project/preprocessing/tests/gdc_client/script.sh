#!/bin/bash

# Download dir
DOWNLOAD_DIR="gdc_download"
# Renamed slides dir
TARGET_DIR="wsi"

# Create the dirs if does not exist
mkdir -p "$TARGET_DIR"
mkdir -p "$DOWNLOAD_DIR"

# Download the data
./gdc-client download -m gdc_manifest.2024-12-15.111244.txt -d "$DOWNLOAD_DIR"

# Rename all .svs
count=1
find "$DOWNLOAD_DIR" -type f -name "*.svs" | while read -r file; do
  new_name="slide_${count}.svs"
  mv "$file" "$TARGET_DIR/$new_name"
  ((count++))
done

# Remove DOWNLOAD_DIR
rm -rf "$DOWNLOAD_DIR"
echo "Done!"
