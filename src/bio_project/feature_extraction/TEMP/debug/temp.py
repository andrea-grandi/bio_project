# @title Load Libraries
from torch_geometric.data import Dataset
import glob
from torch_geometric.data import data
from torch_geometric.loader import DataLoader
import sys
#sys.path.append("/content/mil4wsi")
#from utilsmil4wsi.test import test
import argparse
#from models import selectModel
from torch.nn import BCEWithLogitsLoss
import torch
import os
#import wandb
from argparse import Namespace
import time
import tqdm

data_root = "/Users/andreagrandi/Developer/bio_project/src/bio_project/feature_extraction/output_feats/feats/processed"


# @title Dataset
class CustomDataset(Dataset):
    def __init__(self,root,data_type):
        self.path=os.path.join(root,data_type,"*")
        self.data=glob.glob(self.path)
        #print(self.data)
        self.slides=[torch.load(self.data[idx]) for idx in range(len(self.data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.slides[idx]
        return sample
    
if __name__ == "__main__":
    train_dataset=CustomDataset(data_root,"train")
    test_dataset=CustomDataset(data_root,"test")
    train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=True)

    for batch in train_loader:
        print(batch)
        break
    
