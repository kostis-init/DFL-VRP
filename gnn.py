import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GNN(nn.Module):
    def __init__(self, num_edge_features, edges):
        super(GNN, self).__init__()

        self.edge_index = torch.tensor([[edge.node1.id, edge.node2.id] for edge in edges],
                                       dtype=torch.long).t().contiguous()
        self.conv1 = GCNConv(num_edge_features, 1)
        # self.conv2 = GCNConv(32, 1)

    def forward(self, edge_features):
        x = self.conv1(edge_features, self.edge_index)
        # x = torch.relu(x)
        # x = self.conv2(x, self.edge_index)
        # x = torch.sigmoid(x)
        return x.squeeze()
