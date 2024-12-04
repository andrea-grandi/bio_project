from tifffile import imread, imsave
import glob
import os


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

orig_tile_dir_name = "/Users/andreagrandi/Developer/bio_project/src/bio_project/data/images/saved_tiles/original_tiles/"
output_dir_blank = "/Users/andreagrandi/Developer/bio_project/src/bio_project/data/images/saved_tiles/original_tiles/blank"
output_dir_partial = "/Users/andreagrandi/Developer/bio_project/src/bio_project/data/images/saved_tiles/original_tiles/partial/"
output_dir_good = "/Users/andreagrandi/Developer/bio_project/src/bio_project/data/images/saved_tiles/original_tiles/good/"

img_list = glob.glob(os.path.join(orig_tile_dir_name, "*.tif"))

find_mean_std_pixel_value(img_list, output_dir_blank, output_dir_good, output_dir_partial)
