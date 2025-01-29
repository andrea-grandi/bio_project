import torch
from sklearn.preprocessing import MinMaxScaler

from models.utils.basemodel import Baseline
from models.utils.buffer import FCLayer, BClassifierBuffer, MILNetBuffer, init


class CustomBuffermil(Baseline):
    def __init__(self, args, state_dict_weights):
        super(CustomBuffermil, self).__init__(args, state_dict_weights)
        
        milfc, milbag = FCLayer(self.c_in, self.classes), BClassifierBuffer(self.c_in, self.classes)
        self.mil = MILNetBuffer(milfc, milbag)
        self.mil = init(self.mil, self.state_dict_weights)
        self.inference = False
        self.args = args
        self.aggregationtype = args.bufferaggregate

    def forward_mil(self, feats, results, inference):
        # Second step: MIL
        results["higher"] = self.mil(feats, inference)  # x5x20
        return results

    def preloop(self, epoch, loader):
        if epoch % self.args.buffer_freq == self.args.buffer_freq - 1:
            self.storebuffer(loader)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, levels: torch.Tensor, childof: torch.Tensor, edge_index2: torch.Tensor = None, edge_index3: torch.Tensor = None):
        """Forward model

        Args:
            x (torch.Tensor): input
            edge_index (torch.Tensor): adjacency matrix
            levels (torch.Tensor): scale level
            childof (torch.Tensor): interscale information
            edge_index2 (torch.Tensor, optional): adjacency matrix of single scale
            edge_index3 (torch.Tensor, optional): adjacency matrix of single scale

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
            
            # Extract additional features (cellpose)
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
                results = self(x, edge_index, level, childof, edge_index2, edge_index3)
                pred = torch.sigmoid(results["higher"][1]).squeeze()
                if (pred > 0.2) & (y == 1):
                    A = results["higher"][2]
                    A = MinMaxScaler().fit_transform(A.reshape(-1, 1).cpu().detach().numpy()).reshape(-1)
                    
                    # Normalize the additional features
                    num_cells_norm = MinMaxScaler().fit_transform(num_cells.unsqueeze(1).cpu().detach().numpy()).reshape(-1)
                    cell_density_norm = MinMaxScaler().fit_transform(cell_density.unsqueeze(1).cpu().detach().numpy()).reshape(-1)
                    mean_cell_area_norm = MinMaxScaler().fit_transform(mean_cell_area.unsqueeze(1).cpu().detach().numpy()).reshape(-1)
                    
                    # Combine the features with A, e.g., using a weighted sum
                    # We can adjust the weights as needed
                    weights = torch.tensor([0.5, 0.25, 0.25]).cpu()
                    combined_score = weights[0] * A + weights[1] * num_cells_norm + weights[2] * mean_cell_area_norm
                    
                    self.storeBuffer(x, combined_score, self.args.ntop)

        self.inference = True

    def storeBuffer(self, feats, A, k):
        _, m_indices = torch.sort(A.clone().detach().cuda(), 0, descending=True)
        #_, m_indices = torch.sort(torch.tensor(A).cuda(), 0, descending=True)  # sort scores (A), m_indices in shape N
        m_feats = torch.index_select(feats, dim=0, index=m_indices[:k])  # select top k features
        self.mil.store(m_feats)

    def storeBufferRandom(self, feats, k):
        perm = torch.randperm(feats.size(0))
        idx = perm[:k]
        m_feats = feats[idx]
        self.mil.store(m_feats)

    def bufferinference(self, feats):
        self.mil.bufferinference(feats)


        
