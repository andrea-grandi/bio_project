import torch
import argparse
import joblib
import os
import glob
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from scipy.sparse import coo_matrix

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='prepare the dataset for pytorch geometric')
parser.add_argument('--source', type=str, help='origin folder')
parser.add_argument('--dest', type=str, help='destination folder')
parser.add_argument('--levels', type=int, nargs='+', default=[3], help='destination folder')
args = parser.parse_args()
dest = args.dest
source = args.source
levels = args.levels

def from_scipy_sparse_matrix(A):
    """
    Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.

    Returns:
        edge_index (torch.Tensor): Edge indices tensor of shape (2, num_edges).
        edge_weight (torch.Tensor): Edge attributes tensor of shape (num_edges,).
    """
    A = A.tocoo()
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0).to(torch.long)
    edge_weight = torch.from_numpy(A.data)
    return edge_index, edge_weight


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, type="train"):
        self.type = type 
        self.root = root 
        self._processed_dir = os.path.join(self.root, "processed") 
        os.makedirs(self._processed_dir, exist_ok=True) 

        self.processed_files = glob.glob(os.path.join(self._processed_dir, self.type, "data_*.pt"))
        self.lenght = len(self.processed_files)

        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        Return the raw file names.
        """
        return ['some_file_1', 'some_file_2'] 

    @property
    def processed_file_names(self):
        """
        Return the processed file names.
        """
        return [os.path.basename(f) for f in self.processed_files]

    def len(self):
        """
        Return the length of the dataset.
        """
        return self.lenght

    def process(self):
        """
        Process the dataset.
        """
        bags = glob.glob(os.path.join(source, "*/*"))
        totLevels = levels
        for idx, bag in enumerate(bags):
            try:
                # "train" or "test"
                if "test" in bag:
                    data_type = "test"
                else:
                    data_type = "train"
    
                patches = joblib.load(os.path.join(bag, "embeddings.joblib"))
                patch_level = patches["level"]
                patch_childof = patches["childof"]
                if "label" in patches.columns:
                    patch_label = patches["label"]
                else:
                    patch_label = [-1]
                patch_childof[patch_childof.isnull()] = -1
                embeddings = patches["embedding"]
                size = embeddings.shape[0]
    
                # Cellpose features
                num_cells = patches["num_cells"] if "num_cells" in patches.columns else torch.zeros(size)
                cell_density = patches["cell_density"] if "cell_density" in patches.columns else torch.zeros(size)
                mean_cell_area = patches["mean_cell_area"] if "mean_cell_area" in patches.columns else torch.zeros(size)
    
                x = [torch.Tensor(np.matrix(embeddings[i])) for i in range(size)]
                X = torch.vstack(x)
    
                matrix_adj = torch.load(os.path.join(bag, "adj.th"))
    
                matrix_edges = coo_matrix(matrix_adj)
                edge_index, edge_weight = from_scipy_sparse_matrix(matrix_edges)
    
                matrix_adj_filtered = matrix_adj.clone()
                max_level = np.max(patch_level)
                min_level = np.min(patch_level)
                for level in range(max_level, min_level, -1):
                    matrix_adj_filtered[(patch_level.to_numpy() == level).nonzero()[0], 
                                        (patch_level.to_numpy() == level-1).nonzero()[0].reshape(-1, 1)] = 0
                    matrix_adj_filtered[(patch_level.to_numpy() == level-1).nonzero()[0], 
                                        (patch_level.to_numpy() == level).nonzero()[0].reshape(-1, 1)] = 0
                # Convert to COO matrix
                matrix_edges_filtered = coo_matrix(matrix_adj_filtered)
                edge_index_filtered, _ = from_scipy_sparse_matrix(matrix_edges_filtered)

                data = Data(
                    x=X,
                    edge_index=edge_index,
                    edge_index_filtered=edge_index_filtered,
                    childof=torch.LongTensor(patch_childof),
                    level=torch.LongTensor(patch_level),
                    y=torch.LongTensor([int(os.path.basename(bag).split("_")[-1])]),
                    name=os.path.basename(bag),
                    patch_label=torch.LongTensor(patch_label),
                    x_coord=torch.LongTensor(patches["x"]),
                    y_coord=torch.LongTensor(patches["y"]),
                    num_cells=torch.FloatTensor(num_cells),
                    cell_density=torch.FloatTensor(cell_density),
                    mean_cell_area=torch.FloatTensor(mean_cell_area)
                )
    
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
    
                os.makedirs(os.path.join(self._processed_dir, data_type), exist_ok=True)
                torch.save(data, os.path.join(self._processed_dir, data_type, f'data_{idx}.pt'))

            except Exception as e:
                print(f"Error during processing {bag}: {e}")

    def get(self, idx):
        """
        Return the idx-th data object.
        """
        data_path = os.path.join(self._processed_dir, self.type, f'data_{idx}.pt')
        return torch.load(data_path)


def prepareslide():
    """
    Prepere the slide for the dataset
    """
    global levels
    global dest
    global source
    dataset = MyOwnDataset(root=dest, type="train")
    print(dataset)


if __name__ == '__main__':
    prepareslide() 