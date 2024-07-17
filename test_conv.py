import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np

from sw_gnn import SW_conv

num_nodes = 100
feature_dim = 50
out_dim = 35
edge_prob = 0.2

device = 'cuda'
dtype = torch.float64

# Create a random graph using NetworkX
G = nx.erdos_renyi_graph(num_nodes, edge_prob)

# Extract the edge index and convert it to a PyTorch tensor
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

# Create node features (for simplicity, we use a feature vector of ones)
node_features = torch.randn((num_nodes, feature_dim), dtype=dtype)

# Create a PyTorch Geometric data object
data = Data(x=node_features, edge_index=edge_index)

conv = SW_conv(feature_dim, out_dim, mlp_layers=3, bias=False, concat_self = True, batchNorm_final=True, device=device,dtype=dtype)

conv.eval()

node_features = node_features.to(device=device,dtype=dtype)
edge_index = edge_index.to(device=device)

# Disable gradient computation
with torch.no_grad():  
    # Apply one iteration of SW message passing
    out = conv(node_features, edge_index)
    out2 = conv(16*node_features, edge_index)


print('Relative homogeneity error: ', torch.norm(out2-16*out).item() / torch.norm(out).item())

