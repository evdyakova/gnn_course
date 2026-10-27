
import torch
from torch import nn
from torch_geometric.utils import degree, add_remaining_self_loops

class MyGCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        n = x.size(0)
        # петли и нормировка
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=n)
        row, col = edge_index                  
        deg = degree(col, n, dtype=x.dtype).clamp(min=1)
        # [n, out_dim]
        h = self.lin(x)                            

        norm = deg[row].pow(-0.5) * deg[col].pow(-0.5)                               

        # message + aggregate
        msg = h[row] * norm.unsqueeze(-1)                                   

        out = torch.zeros_like(h)     

        return out.index_add_(0, col, msg)                              

class MySAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
  
    def forward(self, x, edge_index):
        n = x.size(0)
        row, col = edge_index

        # суммируем признаки соседей
        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, col, x[row])

        # число входящих соседей каждой вершины
        deg = degree(col, n, dtype=x.dtype).clamp(min=1)

        # среднее признаков соседей
        neigh_mean = neigh_sum / deg.unsqueeze(-1)

        out = self.lin_neigh(neigh_mean) + self.lin_self(x)

        return out
