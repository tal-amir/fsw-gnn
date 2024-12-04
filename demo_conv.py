# Simple example of FSW-GNN convolutional layer using random graphs

import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np

from fsw_conv import FSW_conv

num_nodes = 100
vertex_feature_dim = 50
edge_feature_dim = 11
out_dim = 35

edge_prob = 0.2 # Edge probability for random graphs

device = 'cuda'
dtype = torch.float32

# Create a random graph using NetworkX
G = nx.erdos_renyi_graph(num_nodes, edge_prob)

# Extract the edge index and convert it to a PyTorch tensor
edge_index = torch.tensor(list(G.edges), dtype=torch.long, device=device).t().contiguous()
num_edges = edge_index.shape[1]

# Create node features (for simplicity, we use a feature vector of ones)
node_features = torch.randn((num_nodes, vertex_feature_dim), dtype=dtype, device=device)
edge_features = None if edge_feature_dim == 0 else torch.randn((num_edges, edge_feature_dim), dtype=dtype, device=device)

node_features = node_features.to(device=device,dtype=dtype)
edge_index = edge_index.to(device=device)

# Create a PyTorch Geometric data object
data = Data(x=node_features, edge_index=edge_index)

conv = FSW_conv(vertex_feature_dim, out_dim, edgefeat_dim=edge_feature_dim, mlp_layers=3, 
                learnable_embedding = True, device=device, dtype=dtype)

# Apply one iteration of FSW message passing
out = conv(node_features, edge_index=edge_index, edge_features=edge_features)


