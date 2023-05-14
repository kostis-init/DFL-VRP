import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_max_pool

import torch.nn.functional as F

from domain.vrp_edge import VRPEdge
from domain.vrp_node import VRPNode
from util import parse_datafile
import numpy as np


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


def graph_to_edge_index(edges: [VRPEdge]):
    # This function takes a list of edges and does a complex thing.
    # First it implicitly swaps the edges to nodes.
    # Secondly, it will create the index matrix but for the nodes which were previously edges.
    # This index matrix will represent the connected edges. An edge is connected to another edge if
    # the first's edge start node is the same as the second's edge end node.
    # Finally, it will return the index matrix.
    # Example:
    # edges = [0 -> 1, 0 -> 2, 1 -> 0, 1 -> 2, 2 -> 0, 2 -> 1]
    # that will be implicitly swapped to nodes:
    # nodes = [0, 1, 2, 3, 4, 5]
    # e.g. node 0 is edge 0 -> 1 and is connected to edge 1 -> 0, 2 -> 0, and 1 -> 2, so 0 is connected to nodes 2, 3, 4
    # e.g. node 1 is edge 0 -> 2 and is connected to edge 1 -> 0, 2 -> 0, and 2 -> 1, so 1 is connected to nodes 2, 4, 5
    # e.g. node 2 is edge 1 -> 0 and is connected to edge 0 -> 1, 2 -> 1, and 0 -> 2, so 2 is connected to nodes 0, 1, 5
    # e.g. node 3 is edge 1 -> 2 and is connected to edge 0 -> 1, 2 -> 1, and 2 -> 0, so 3 is connected to nodes 0, 4, 5
    # e.g. node 4 is edge 2 -> 0 and is connected to edge 0 -> 2, 1 -> 2, and 0 -> 1, so 4 is connected to nodes 0, 1, 3
    # e.g. node 5 is edge 2 -> 1 and is connected to edge 0 -> 2, 1 -> 2, and 1 -> 0, so 5 is connected to nodes 1, 2, 3
    # The index matrix will look like this:
    # [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
    #  [2, 3, 4, 2, 4, 5, 0, 1, 5, 0, 4, 5, 0, 1, 3, 1, 2, 3]]
    # The first row represents the start nodes and the second row represents the end nodes.

    # So, for each edge find the edges indices that have the same start node as the current edge's end node
    # or the same end node as the current edge's start node.
    # This will be the index matrix.
    # The index matrix will be a 2xN
    index_matrix = []
    for i1, e1 in enumerate(edges):
        # find the edges that have the same start node as the current edge's end node
        # or the same end node as the current edge's start node.
        for i2, e2 in enumerate(edges):
            if e2.node1 == e1.node2 or e2.node2 == e1.node1:
                # add i1, i2 to the index matrix
                index_matrix.append([i1, i2])

    return index_matrix


class GATModel(nn.Module):
    def __init__(self, num_edge_features, edges):
        super(GATModel, self).__init__()

        self.edge_index = torch.tensor(graph_to_edge_index(edges), dtype=torch.long).t().contiguous()
        self.conv1 = GATConv(num_edge_features, 32, heads=4)
        self.conv2 = GATConv(32*4, 1, heads=1)


    def forward(self, edge_features):
        x = F.relu(self.conv1(edge_features, self.edge_index))
        x = self.conv2(x, self.edge_index)
        return x.squeeze()


class EdgeGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EdgeGNN, self).__init__()
        self.message_function = nn.Linear(input_dim, hidden_dim)
        self.aggregation_function = nn.Linear(hidden_dim, output_dim)

    def forward(self, edge_features, adjacency_matrix):
        # edge_features: tensor of shape (num_edges, input_dim)
        # adjacency_matrix: binary tensor of shape (num_edges, num_edges)

        # Compute messages
        messages = self.message_function(edge_features)  # (num_edges, hidden_dim)

        # Aggregate messages from neighboring edges
        aggregated_messages = torch.matmul(adjacency_matrix, messages)  # (num_edges, hidden_dim)

        # Apply aggregation function
        new_edge_features = self.aggregation_function(aggregated_messages)  # (num_edges, output_dim)

        return new_edge_features

# vrp = parse_datafile('data/cvrp_10000_25_4_8_0.1/instance_0')
# num_nodes = len(vrp.nodes)
# num_edges = len(vrp.edges)
# num_feat = len(vrp.edges[0].features)
#
# # Example usage
# # input_dim = 4
# # hidden_dim = 64
# # output_dim = 1  # Predicting a single scalar (edge cost)
#
# # Creating a random adjacency matrix (for demonstration purposes)
# # adjacency_matrix = torch.randint(0, 2, (num_edges, num_edges))
#
# # Adjacency matrix for the VRp, should be a fully connected graph except for self loops
# adjacency_matrix = torch.ones((num_edges, num_edges)) - torch.eye(num_edges)
#
# # Get edge_features tensor, should be of shape (num_edges, num_feat)
# edge_features = torch.tensor([edge.features for edge in vrp.edges])
#
# # Instantiate the EdgeGNN model
# model = EdgeGNN(num_feat, 64, 1)
#
# # Forward pass
# predicted_edge_costs = model(edge_features, adjacency_matrix)
# print(predicted_edge_costs.shape)  # Output shape: (num_edges, output_dim)
