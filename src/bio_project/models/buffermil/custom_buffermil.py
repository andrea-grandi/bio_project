import torch
from sklearn.preprocessing import MinMaxScaler

from models.utils.basemodel import Baseline
from models.utils.buffer import FCLayer, BClassifierBuffer, MILNetBuffer, init


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
        results["higher"] = self.mil(feats, inference)
        return results

    def preloop(self, epoch, loader):
        if epoch % self.args.buffer_freq == self.args.buffer_freq - 1:
            self.storebuffer(loader)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, levels: torch.Tensor, childof: torch.Tensor, edge_index2: torch.Tensor=None, edge_index3: torch.Tensor=None):
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

    def storeBuffer(self, feats, A, k, patch_names, cell_counts, cell_threshold=50):
        A_tensor = torch.Tensor(A).cuda()
        cell_weights = torch.ones_like(A_tensor)
        for i, patch_name in enumerate(patch_names):
            cell_count = cell_counts.get(patch_name, 0)
            if cell_count > cell_threshold:
                cell_weights[i] = 1.5
            else:
                cell_weights[i] = 0.5
        modified_A = A_tensor * cell_weights
        _, m_indices = torch.sort(modified_A, 0, descending=True)
        m_feats = torch.index_select(feats, dim=0, index=m_indices[:k])
        self.mil.store(m_feats)

    def storeBufferRandom(self, feats, k):
        perm = torch.randperm(feats.size(0))
        idx = perm[:k]
        m_feats = feats[idx]
        self.mil.store(m_feats)