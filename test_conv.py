import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np

from fsw_conv import FSW_conv

num_nodes = 100
vertex_feature_dim = 50
edge_feature_dim = 11
out_dim = 35
edge_prob = 0.2

is_homogeneous = True
squeeze_edge_features_when_possible = True # squeezes the edge features' last dimension and makes them scalar when edge_feature_dim = 1
vertex_degree_encoding_function = 'log'

test_grad = True
test_homogeneity = True

device = 'cuda'
dtype = torch.float64

# Create a random graph using NetworkX
G = nx.erdos_renyi_graph(num_nodes, edge_prob)

# Extract the edge index and convert it to a PyTorch tensor
edge_index = torch.tensor(list(G.edges), dtype=torch.long, device=device).t().contiguous()
num_edges = edge_index.shape[1]

# Create node features (for simplicity, we use a feature vector of ones)
node_features = torch.randn((num_nodes, vertex_feature_dim), dtype=dtype, device=device, requires_grad=test_grad)
edge_features = None if edge_feature_dim == 0 else torch.randn((num_edges, edge_feature_dim), dtype=dtype, device=device, requires_grad=test_grad)

if squeeze_edge_features_when_possible and (edge_feature_dim==1):
    edge_features = edge_features.squeeze(-1)

# Create a PyTorch Geometric data object
data = Data(x=node_features, edge_index=edge_index)

conv = FSW_conv(vertex_feature_dim, out_dim, edgefeat_dim=edge_feature_dim, mlp_layers=3, bias=not is_homogeneous,
                vertex_degree_encoding_function=vertex_degree_encoding_function, 
                vertex_degree_encoding_scale=10,
                homog_degree_encoding=is_homogeneous, 
                learnable_embedding = True,
                concat_self = True, batchNorm_final=True, device=device, dtype=dtype, self_loop_weight=0.2)

conv.eval()

node_features = node_features.to(device=device,dtype=dtype)
edge_index = edge_index.to(device=device)

# Apply one iteration of SW message passing
out = conv(node_features, edge_index=edge_index, edge_features=edge_features)
out2 = conv(16*node_features, edge_index=edge_index, edge_features=16*edge_features)

print('')
print('Forward pass went ok')

if test_grad:
    obj = node_features.norm() + edge_features.norm()
    obj.backward()
    print('Backward pass went ok')

if test_homogeneity:
    if is_homogeneous:
        print('Relative deviation from homogeneity: ', torch.norm(out2-16*out).item() / torch.norm(out).item())
    else:
        print('Relative deviation from homogeneity: ', torch.norm(out2-16*out).item() / torch.norm(out).item(), '(this is ok since is_homogeneous=False)')

