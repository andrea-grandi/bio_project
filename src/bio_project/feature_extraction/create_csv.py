import pandas as pd
import glob
import os

listanames = glob.glob("/Users/andreagrandi/Developer/bio_project_temp/slides/*")
listanames = [name.split("/")[-1] for name in listanames]
listanames = [name.split(".")[-2] for name in listanames]

labels=[int(name.split("_")[-1]=="tumor") for name in listanames]

types=["test" if name.split("_")[0]=="test" else "train" for name in listanames]

pd.DataFrame({"image":listanames,"label":labels,"phase":types}).to_csv("/Users/andreagrandi/Developer/bio_project_temp/feature_extraction/cam_multi.csv",index=False)
