import roboflow

rf = roboflow.Roboflow(api_key='A1HqGsd7n9GXR5ZyckOQ')

# get a workspace
workspace = rf.workspace("https://app.roboflow.com/andrea-grandi/camelyon_segmentation/upload")

#https://app.roboflow.com/andrea-grandi/camelyon_segmentation/upload

# Upload data set to a new/existing project
workspace.upload_dataset(
    "COCO_camelyon_without_annotations/", # This is your dataset path
    "camelyon_segmentation", # This will either create or get a dataset with the given ID
    num_workers=10,
    project_license="MIT",
    project_type="object-detection",
    batch_name=None,
    num_retries=0
)
