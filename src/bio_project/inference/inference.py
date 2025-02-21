from torch_geometric.data import Dataset
import glob
from torch_geometric.data import data
from torch_geometric.loader import DataLoader
import sys
import argparse
import wandb
from torch.nn import BCEWithLogitsLoss
import torch
import os
import wandb
from argparse import Namespace
import time
import tqdm
import pandas as pd

from utils.test import test
from models import selectModel


data_root="/Users/andreagrandi/Developer/bio_project/src/bio_project/inference/final_embeddings/processed"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self,root,data_type):
        self.path=os.path.join(root,data_type,"*")
        self.data=glob.glob(self.path)
        self.slides=[torch.load(self.data[idx]) for idx in range(len(self.data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.slides[idx]
        return sample
    

def get_args():
    parser = argparse.ArgumentParser()

    # Optimization arguments
    group1 = parser.add_argument_group("optimization")
    group1.add_argument('--lr', default=0.001,
                        type=float, help='learning rate')
    group1.add_argument('--weight_decay', default=0.0001,
                        type=float, help='Weight decay [5e-3]')

    # GNN arguments
    group2 = parser.add_argument_group("gnn")
    group2.add_argument('--residual', default=False, action="store_true",)
    group2.add_argument('--num_layers', default=1, type=int,
                        help='number of Graph layers')
    group2.add_argument('--dropout', default=True, action="store_true")
    group2.add_argument('--dropout_rate', default=0.2, type=float)
    group2.add_argument('--layer_name', default="GAT",
                        type=str, help='layer graph name')
    group2.add_argument('--heads', default=2, type=int,
                        help='layer graph name')

    # Training arguments
    group3 = parser.add_argument_group("training")
    group3.add_argument('--seed', default=12, type=int,
                        help='seed for reproducibility')
    group3.add_argument('--n_epoch', default=100,
                        type=int, help='number of epochs')

    # Dimensions arguments
    group4 = parser.add_argument_group("dimensions")
    group4.add_argument('--n_classes', default=1, type=int,
                        help='Number of output classes [2]')
    group4.add_argument('--c_hidden', default=256,
                        type=int, help='intermediate size ')
    group4.add_argument('--input_size', default=384,
                        type=int, help='input size ')

    # Dataset arguments
    group5 = parser.add_argument_group("dataset")
    group5.add_argument('--scale', default="0", type=str,
                        help='scale resolution')
    group5.add_argument('--dataset', default="cam", type=str,
                        choices=["cam", "lung"], help='input size ')
    group5.add_argument('--datasetpath',  type=str, help='dataset path')

    # Distillation arguments
    group6 = parser.add_argument_group("distillation")
    group6.add_argument('--lamb', default=1, type=float, help='lambda')
    group6.add_argument('--beta', default=1, type=float, help='beta')
    group6.add_argument('--temperature', default=1.5, type=float, help='temperature')
    group6.add_argument('--add_bias', default=True, action="store_true")
    group6.add_argument('--max', default=True, action="store_true")
    group6.add_argument('--checkpoint', default="/Users/andreagrandi/Developer/bio_project/weights/model/checkpoint.pt", type=str, help='checkpoint')

    parser.add_argument('--tag', default="split", type=str, help='train strategy')
    parser.add_argument('--modeltype', default="CustomBuffermil", type=str, help='train strategy')
    parser.add_argument('--project', default="bio_project", type=str, help='project name for wandb')
    parser.add_argument('--model', default="bio_project", type=str, help='project name for wandb')
    parser.add_argument('--wandbname', default="main", type=str, help='project name for wandb')

    group7 = parser.add_argument_group("submitit")
    group7.add_argument('--partition', default="prod",type=str,help='partition name')
    group7.add_argument('--time', default=120, type=float, help='job duration')
    group7.add_argument('--nodes', default=1, type=int, help='number of jobs')
    group7.add_argument('--job_name', default="dasmil",type=str,help="job name")
    group7.add_argument('--mem', default=32, type=int, help='ram requested GB')
    group7.add_argument('--job_parallel', default=10, type=int, help='number of jobs in parallel')
    group7.add_argument('--logfolder', default="logfolder", type=str, help='log folder location name')

    # Buffermil parameters
    group8 = parser.add_argument_group("submitit")
    group8.add_argument("--randomstore", default=False, help="ramdom sampling during the buffer storage")
    group8.add_argument("--bufferaggregate", default="mean", choices=["mean","max","diffmax"], help="type of buffer aggregation")
    group8.add_argument("--ntop", default=10, help="number of patches stored in the buffer per each image")
    group8.add_argument('--buffer_freq', default=10, type=int, help='frequency to update the buffer')
    
    filtered_args = [arg for arg in sys.argv if not arg.startswith('-f')]
    args = parser.parse_args(filtered_args[2:])
    return args
  

def run_inference(model, test_loader):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for data in tqdm.tqdm(test_loader, desc="Inference"):
            data = data.to(device)
            x, edge_index, childof, level = data.x, data.edge_index, data.childof, data.level

            edge_index2 = data.edge_index_2 if hasattr(data, 'edge_index_2') else None
            edge_index3 = data.edge_index_3 if hasattr(data, 'edge_index_3') else None

            output = model(x, edge_index, level, childof, edge_index2, edge_index3)
            pred = torch.sigmoid(output["higher"][1]).cpu().numpy() 

            predictions.append(pred)
            labels.append(data.y.cpu().numpy())

    return predictions, labels


def inference():  
  test_dataset = CustomDataset(data_root, "test")
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
  
  """
  for i in test_loader:
    print(i)
    break
  """
  args = get_args()

  model = selectModel(args)
  model.kl = None
  model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device(device)))
  model.to(device)

  predictions, labels = run_inference(model, test_loader)

  # Save results
  df = pd.DataFrame({"Prediction": predictions, "Label": labels})
  df.to_csv("inference_results.csv", index=False)

  print("✅ Inference completed. Results saved into: 'inference_results.csv'.")
  print(f"🚀 Label: {labels[1]}")
