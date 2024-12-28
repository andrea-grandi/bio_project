from tifffile import imread, imsave
import glob
import os
import argparse


def find_mean_std_pixel_value(img_list, output_dir_blank, output_dir_good, output_dir_partial):
    for file in img_list:
        img = imread(file)
        avg = img.mean()
        std = img.std()
        
        file_name = os.path.basename(file)
        save_path = None
        
        if avg < 180 and std > 30:
            save_path = os.path.join(output_dir_good, file_name)
        elif avg > 230 and std < 30:
            save_path = os.path.join(output_dir_blank, file_name)
        else:
            save_path = os.path.join(output_dir_partial, file_name)
            
        imsave(save_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_tile_dir_name", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/data/images/saved_tiles/original_tiles/")
    parser.add_argument("--output_dir_blank", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/data/images/saved_tiles/original_tiles/blank")
    parser.add_argument("--output_dir_partial", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/data/images/saved_tiles/original_tiles/partial/")
    parser.add_argument("--output_dir_good", type=str, default="/Users/andreagrandi/Developer/bio_project/src/bio_project/data/images/saved_tiles/original_tiles/good/")
    args = parser.parse_args()
    
    orig_tile_dir_name = args.orig_tile_dir_name
    output_dir_blank = args.output_dir_blank
    output_dir_partial = args.output_dir_partial
    output_dir_good = args.output_dir_good
    
    img_list = glob.glob(os.path.join(orig_tile_dir_name, "*.tif"))
    
    find_mean_std_pixel_value(img_list, output_dir_blank, output_dir_good, output_dir_partial)