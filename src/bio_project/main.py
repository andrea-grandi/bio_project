import os
import torch
import yaml
import glob

from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from utils.experiments import *
from utils.parser import get_args
from utils.training import train
from models import selectModel

# Set environment variable to increase wandb service wait time
os.environ["WANDB__SERVICE_WAIT"] = "300"

# sys.path.append('.')

# Ensure that all operations are deterministic on GPU (if used) for reproducibility

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
    

def load_config(config_path):
   """Loading config.yaml file"""
   with open(config_path, "r") as file:
      config = yaml.safe_load(file)
   return config

def main():
    # Get command line arguments
    args = get_args()
    
    # Load configurations
    config = load_config("config_train.yaml")

    train_dataset=CustomDataset(config["data_root"],"train")
    test_dataset=CustomDataset(config["data_root"],"test")
    train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=True)

    model = selectModel(args)
    model.kl = None

    train(model=model, train_loader=train_loader, test_loader=test_loader, args=args)
    
    """
    executor = submitit.AutoExecutor(folder=args.logfolder, slurm_max_num_timeout=30)
    executor.update_parameters(
            mem_gb=args.mem,
            slurm_gpus_per_task=args.nodes,
            tasks_per_node=args.nodes,  # one task per GPU
            slurm_cpus_per_gpu=args.nodes,
            nodes=args.nodes,
            timeout_min=args.time,  # max is 60 * 72
            # Below are cluster dependent parameters
            slurm_partition=args.partition,
            slurm_signal_delay_s=120,
            slurm_array_parallelism=args.job_parallel)
    executor.update_parameters(name=args.job_name)
    experiments=[]
    experiments=experiments+launch_buffermil(args)
    
    executor.map_array(processDataset,experiments)
    
    #processDataset(experiments[0])
    """

if __name__ == '__main__':
    main()
