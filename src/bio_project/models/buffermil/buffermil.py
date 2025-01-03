"""
@inproceedings{Bontempo2023_MICCAI,
    author={Bontempo, Gianpaolo and Porrello, Angelo and Bolelli, Federico and Calderara, Simone and Ficarra, Elisa},
    title={{DAS-MIL: Distilling Across Scales for MIL Classification of Histological WSIs}},
    booktitle={Medical Image Computing and Computer Assisted Intervention – MICCAI 2023},
    pages={248--258},
    year=2023,
    month={Oct},
    publisher={Springer},
    doi={https://doi.org/10.1007/978-3-031-43907-0_24},
    isbn={978-3-031-43906-3}
}
"""
import torch
from sklearn.preprocessing import MinMaxScaler

from utils.dropout import dropout_node
from models.utils.basemodel import Baseline
from models.utils.modules import FCLayer, BClassifierBuffer, MILNetBuffer, init


class Buffermil(Baseline):
    def __init__(self, args, state_dict_weights):
        super(Buffermil, self).__init__(args, state_dict_weights)

        milfc, milbag = FCLayer(self.c_in, self.classes), BClassifierBuffer(self.c_in, self.classes)
        self.mil = MILNetBuffer(milfc, milbag)
        self.mil = init(self.mil, self.state_dict_weights)
        self.inference = False
        self.args = args
        self.aggregationtype = args.bufferaggregate

    def forward_mil(self, feats, results, inference):
        #second step: MIL
        results["higher"] = self.mil(feats, inference)#x5x20
        return results

    def preloop(self, epoch, loader):
        if epoch % self.args.buffer_freq == self.args.buffer_freq - 1:
            self.storebuffer(loader)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, levels: torch.Tensor, childof: torch.Tensor, edge_index2: torch.Tensor=None, edge_index3: torch.Tensor=None):
        """forward model

        Args:
            x (torch.Tensor): input
            edge_index (torch.Tensor): adjecency matrix
            levels (torch.Tensor): scale level
            childof (torch.Tensor): interscale information
            edge_index2 (torch.Tensor, optional): adjecency matrix of single scale
            edge_index3 (torch.Tensor, optional): adjecency matrix of single scale

        Returns:
            _type_: _description_
        """
        results = {}
        if self.inference:
            results["higher"] = self.mil.bufferinference(x, self.inference, self.aggregationtype)
        else:
            results = self.forward_mil(x, results, self.inference)
        return results

    def storebuffer(self, loader):
        self.mil.buffer = None
        self.inference = False
        for _, data in enumerate(loader):
            data = data.cuda()
            x, edge_index, childof, level, y = data.x, data.edge_index, data.childof, data.level, data.y
            if data.__contains__("edge_index_2") and data.__contains__("edge_index_3"):
                edge_index2, edge_index3 = data.edge_index_2, data.edge_index_3
            else:
                edge_index2 = None
                edge_index3 = None
            if self.args.randomstore:
                self.storeBufferRandom(x, self.args.ntop)
            else:
                results = self(x, edge_index, level, childof, edge_index2, edge_index3)
                pred = torch.sigmoid(results["higher"][1]).squeeze()
                if (pred > 0.2) & (y == 1):
                    A = results["higher"][2]
                    A = MinMaxScaler().fit_transform(A.reshape(-1, 1).cpu().detach().numpy()).reshape(-1)
                    self.storeBuffer(x, A, self.args.ntop)

        self.inference = True

    """
    Critical instance selection and buffer storing
    k critical instances are selected
    """
    def storeBuffer(self, feats, A, k):
        _, m_indices = torch.sort(torch.Tensor(A).cuda(), 0, descending = True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim = 0, index = m_indices[:k]) # select critical instances, m_feats in shape C x K
        self.mil.store(m_feats)

    """
    Custom buffer storing with cell count
    Andrea Grandi and Daniele Vellani
    """
    def storeBuffer(self, feats, A, k, patch_names, cell_counts, cell_threshold=50):
        """
        Store the most critical instances in the buffer considering both attention scores and cell count
        
        Args:
            feats: Feature tensors of patches
            A: Original attention scores
            k: Number of patches to select
            patch_names: List of patch names corresponding to feats
            cell_counts: Dictionary mapping patch names to cell counts
            cell_threshold: Minimum number of cells to consider a patch critical
        """
        # Convert attention scores to tensor
        A_tensor = torch.Tensor(A).cuda()
        
        # Create cell count weights
        cell_weights = torch.ones_like(A_tensor)
        for i, patch_name in enumerate(patch_names):
            cell_count = cell_counts.get(patch_name, 0)
            if cell_count > cell_threshold:
                # Boost attention score for patches with high cell count
                cell_weights[i] = 1.5
            else:
                # Reduce attention score for patches with low cell count
                cell_weights[i] = 0.5
                
        # Modify attention scores based on cell count
        modified_A = A_tensor * cell_weights
        
        # Sort based on modified scores
        _, m_indices = torch.sort(modified_A, 0, descending=True)
        
        # Select critical instances
        m_feats = torch.index_select(feats, dim=0, index=m_indices[:k])
        
        # Store in buffer
        self.mil.store(m_feats)

    """
    In the first itearation a random buffer is stored
    """
    def storeBufferRandom(self, feats, k):
        perm = torch.randperm(feats.size(0))
        idx = perm[:k]
        m_feats = feats[idx]
        self.mil.store(m_feats)

    def bufferinference(self, feats):
        self.mil.bufferinference(feats)