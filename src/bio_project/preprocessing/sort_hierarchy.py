import shutil
import glob
import os
import submitit
import pandas as pd
import argparse


def get_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='sort_slide')
    parser.add_argument('--sourcex20', default="SOURCEPATH20", type=str, help='path to patches at 20x scale')
    parser.add_argument('--slurm_partition', default="SLURM_PARTITION", type=str, help='slurm partition')
    parser.add_argument('--step', default=10, type=int, help='how many slides process within each job')
    parser.add_argument('--dest', default="DESTINATIONPATH", type=str, help='destination folder')
    args = parser.parse_args()
    return args


def nested_patches(candidate, args):
    """
    Process nested patches for level 0 (20x).

    Args:
        candidate (int): Index of the candidate slide.
        args (argparse.Namespace): Parsed command line arguments.
    """
    dest = args.dest
    lista = glob.glob(os.path.join(args.sourcex20, "*"))
    real_name = lista[candidate].split(os.sep)[-1]
    id = real_name
    test = ""
    label = ""

    levelx20path = os.path.join(args.sourcex20, real_name, "0", "*.jpg")
    dest_folder = os.path.join(dest, test + id)# + "_" + str(label))

    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    # Check if the number of patches in the destination matches the source
    if len(glob.glob(os.path.join(dest_folder, "*.jpg"))) == len(glob.glob(levelx20path)):
        return

    # Copy each patch from the source to the destination
    for patch_x20_path in glob.glob(levelx20path):
        patch_name = os.path.basename(patch_x20_path)
        dest_patch_path = os.path.join(dest_folder, patch_name)
        if not os.path.isfile(dest_patch_path):
            shutil.copy(patch_x20_path, dest_patch_path)
            print(f"Copied {patch_name} to {dest_folder}", flush=True)


def prepareslide(candidates, args):
    """
    Prepare slide data for processing.

    Args:
        candidates (list): List of candidate slide indices.
        args (argparse.Namespace): Parsed command line arguments.
    """
    log_folder = "LOGFOLDER/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(slurm_partition=args.slurm_partition,
                               name="sort_hierarchy", slurm_time=200, mem_gb=10, slurm_array_parallelism=3)
    args = [args for _ in candidates]
    executor.map_array(nested_patches, candidates, args)


if __name__ == "__main__":
    args = get_args()
    real_candidates = []
    lista = glob.glob(os.path.join(args.sourcex20, "*"))

    # Identify real candidates for slide processing
    for candidate in range(len(lista)):
        real_name = lista[candidate].split(os.sep)[-1]
        id = real_name
        test = ""
        label = ""

        levelx20path = os.path.join(args.sourcex20, real_name, "0", "*.jpg")
        dest_folder = os.path.join(args.dest, test + id + "_" + str(label))

        # Check if the number of patches in the destination matches the source
        if len(glob.glob(os.path.join(dest_folder, "*.jpg"))) != len(glob.glob(levelx20path)):
            real_candidates.append(candidate)

    # Process the real candidates
    prepareslide(real_candidates, args)