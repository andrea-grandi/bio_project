import pandas as pd

metadata_camelyon_path = "selected_patches_from_camelyon17/metadata.csv"
metadata_seg_path = "selected_patches_from_camelyon17/extracted_metadata.csv"

df1 = pd.read_csv(metadata_camelyon_path)
df2 = pd.read_csv(metadata_seg_path)

df2_copy = df2.copy()
df2_copy = df2_copy.drop(columns=['bounding_box', 'area'])

df2_copy = df2_copy.groupby(list(df2_copy.columns)).size().reset_index(name='counts')
df2_copy = df2_copy.drop(columns='counts')
print(df2_copy.head())


df2_copy.rename(columns={
    "Patient": "patient",
    "Node": "node",
    "X_coord": "x_coord",
    "Y_coord": "y_coord"
    }, inplace=True)

merged_df = pd.merge(df1, df2_copy, on=["patient", "node", "x_coord", "y_coord"], how="left")

# Path to output metadata
output_path = "selected_patches_from_camelyon17/final_metadata.csv"
merged_df.to_csv(output_path, index=False)

print(f"Merged completed. Output file saved in {output_path}")
