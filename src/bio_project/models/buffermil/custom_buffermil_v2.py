import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from models.utils.basemodel import Baseline
from models.utils.buffer import FCLayer, BClassifierBuffer, MILNetBuffer, init

class CustomBuffermilV2(Baseline):
    def __init__(self, args, state_dict_weights):
        super(CustomBuffermilV2, self).__init__(args, state_dict_weights)
        
        milfc, milbag = FCLayer(self.c_in, self.classes), BClassifierBuffer(self.c_in, self.classes)
        self.mil = MILNetBuffer(milfc, milbag)
        self.mil = init(self.mil, self.state_dict_weights)
        self.inference = False
        self.args = args
        self.aggregationtype = args.bufferaggregate
        
        self.gate_layer = nn.Sequential(
            nn.Linear(3, self.c_in),
            nn.Sigmoid()
        )
    
    def forward_mil(self, feats, results, inference):
        results["higher"] = self.mil(feats, inference)  # x5x20 ad esempio
        return results

    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                levels: torch.Tensor, 
                childof: torch.Tensor, 
                edge_index2: torch.Tensor = None, 
                edge_index3: torch.Tensor = None, 
                cellpose_feats: torch.Tensor = None,
                return_feats: bool = False):

        if cellpose_feats is not None:
            # Computes the gates (coefficients between 0 and 1 for each feature) and scales x accordingly
            gate = self.gate_layer(cellpose_feats)  # shape: (N, c_in)
            x = x * gate  # gating: each feature is scaled by its corresponding coefficient
        
        results = {}
        if self.inference:
            results["higher"] = self.mil.bufferinference(x, self.inference, self.aggregationtype)
        else:
            results = self.forward_mil(x, results, self.inference)
        
        if return_feats:
            return results, x
        else:
            return results

    def preloop(self, epoch, loader):
        if epoch % self.args.buffer_freq == self.args.buffer_freq - 1:
            self.storebuffer(loader)

    def storebuffer(self, loader):
        self.mil.buffer = None
        self.inference = False
        for _, data in enumerate(loader):
            data = data.cuda()
            x, edge_index, childof, level, y = data.x, data.edge_index, data.childof, data.level, data.y

            num_cells = data.num_cells
            cell_density = data.cell_density
            mean_cell_area = data.mean_cell_area

            if data.__contains__("edge_index_2") and data.__contains__("edge_index_3"):
                edge_index2, edge_index3 = data.edge_index_2, data.edge_index_3
            else:
                edge_index2 = None
                edge_index3 = None

            if self.args.randomstore:
                self.storeBufferRandom(x, self.args.ntop)
            else:
                # shape (N, 3)
                cellpose_feats = torch.stack([num_cells, cell_density, mean_cell_area], dim=1)
                results, gated_feats = self(x, edge_index, level, childof, edge_index2, edge_index3, 
                                            cellpose_feats=cellpose_feats, return_feats=True)
                pred = torch.sigmoid(results["higher"][1]).squeeze()
                if (pred > 0.2) & (y == 1):
                    A = results["higher"][2]
                    A = MinMaxScaler().fit_transform(A.reshape(-1, 1).cpu().detach().numpy()).reshape(-1)
                    
                    self.storeBuffer(gated_feats, A, self.args.ntop)

        self.inference = True

    def storeBuffer(self, feats, A, k):
        _, m_indices = torch.sort(torch.tensor(A).cuda(), 0, descending=True)  # sort scores (A), m_indices in shape N
        m_feats = torch.index_select(feats, dim=0, index=m_indices[:k])
        self.mil.store(m_feats)

    def storeBufferRandom(self, feats, k):
        perm = torch.randperm(feats.size(0))
        idx = perm[:k]
        m_feats = feats[idx]
        self.mil.store(m_feats)

    def bufferinference(self, feats):
        self.mil.bufferinference(feats)
