# Sliced-Wasserstein GNN
# Part of anonymous NeurIPS 2024 submission titled "bla bla bla"

import numpy as np

import torch
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from sw_embedding import SW_embedding, minimize_mutual_coherence

class SW_conv(MessagePassing):
    # in_channels:    dimension of input vertex features
    #
    # out_channels:   dimension of output vertex features
    #
    # embed_dim:      output dimension of the SW-embedding of neighboring vertex features.
    #                 if <mlp_layers> == 0 and <concat_self> == False, this argument is forced to equal out_channels.
    #                 Default: 2*max(<in_channels>, <out_channels>)  (chosen heuristically)
    #
    # learnable_embedding: tells whether the SW-embedding parameters (slices, frequency) are learnable or fixed.
    #                      default: True
    #
    # concat_self: when set to True, the embedding of each vertex's neighborhood is concatenated to its own
    #              feature vector before it is passed to the MLP. If <mlp_layers> = 0, then dimensionality reduction
    #              is applied after the concatenation to yield a vector of dimension <out_channels>.
    #              note that when set to False, the network is not fully expressive.
    #              Default: True
    #                   
    # self_loop_weight: when set to a positive number, a self loop with this weight is added to each vertex.
    #                   this is necessary in order to avoid empty neighborhoods (since the SW embedding is
    #                   not defiend for empty sets).
    #                   if set to zero and an empty neighborhood is encountered, this will lead to a runtime error.
    #                   Default: 1
    #
    # bias:           if set to true, the MLP uses a bias vector.
    #                 to make the model scale equivariant (i.e. positively homogeneous) with respect to the vertex features, set 
    #                 all of <bias>, <batchNorm_final>, <batchNorm_hidden> to False, and use a scale-equivariant activation
    #                 (e.g. ReLU, or the Leaky ReLU used by deafault).
    #                 Default: True
    #
    # mlp_layers:     number of MLP layers to apply to the embedding that aggregates neighbor vertex features.
    #                 0 - do not apply an MLP; instead pass the embedded neighborhood directly as the output vertex features
    #                 Default: 1
    #
    # mlp_hidden_dim: dimension of the hidden layers in the MLP. only takes effect if mlp_layers > 1.
    #                 Default: max(in_channels, out_channels)  (chosen heuristically)
    #
    # mlp_activation_final, mlp_activation_hidden:
    #                 activation function to be used at the output of the final and hidden MLP layers.
    #                 if set to None, does not apply an activation
    #                 Defaults: Leaky ReLU with a negative slope of 0.2
    #
    # batchNorm_final, batchNorm_hidden:
    #                 Tell whether to apply batch normalization to the outputs of the final and hidden layers.
    #                 The normalization is applied after the linear layer and before the activation.
    #                 Defaults: False
    #
    # dropout_final, dropout_hidden:
    #                 Dropout probabilities 0 <= p < 1 to be used at the final and hidden layers of the MLP.
    #                 The order of each layer is:  Linear transformation -> Batch normalization -> Activation -> Dropout
    #                 Defaults: 0

    def __init__(self,
                 in_channels, out_channels,
                 embed_dim=None, learnable_embedding=True,
                 concat_self = True,
                 bias=True,
                 mlp_layers=1, mlp_hidden_dim=None,
                 mlp_activation_final = torch.nn.LeakyReLU(negative_slope=0.2), 
                 mlp_activation_hidden = torch.nn.LeakyReLU(negative_slope=0.2), 
                 batchNorm_final = False, batchNorm_hidden = False,
                 dropout_final = 0, dropout_hidden = 0,
                 self_loop_weight = 1,
                 device=None, dtype=torch.float32):
        
        super().__init__(aggr=None)

        if mlp_hidden_dim is None:
            mlp_hidden_dim = max(in_channels, out_channels)

        if (mlp_layers == 0) and (concat_self == False):
            embed_dim = out_channels
        elif embed_dim == None:
            embed_dim = 2*max(in_channels, out_channels)

        # If we're using an MLP and bias==True, then the MLP will add a bias anyway.
        embed_bias = (bias and mlp_layers == 0)
        
        self.concat_self = concat_self
        self.self_loop_weight = self_loop_weight

        # if mlp_layers=0, mlp_input_dim is used also to determine the size of the dimensionality-reduction matrix
        if concat_self:
            mlp_input_dim = in_channels + embed_dim
        else:
            mlp_input_dim = embed_dim

        # construct MLP
        if mlp_layers == 0:
            self.mlp = None

            if concat_self:
                with torch.no_grad():
                    dim_reduct = torch.randn(size=(out_channels, mlp_input_dim), device=device, dtype=dtype, requires_grad=False)
                    dim_reduct = minimize_mutual_coherence(dim_reduct, report=False)
                
                self.dim_reduct = torch.nn.Parameter(dim_reduct, requires_grad=learnable_embedding)
        else:
            mlp_modules = []
            
            for i in range(mlp_layers):
                in_curr = mlp_input_dim if i == 0 else mlp_hidden_dim
                out_curr = out_channels if i == mlp_layers-1 else mlp_hidden_dim
                act_curr = mlp_activation_final if i == mlp_layers-1 else mlp_activation_hidden
                bn_curr = batchNorm_final if i == mlp_layers-1 else batchNorm_hidden
                dropout_curr = dropout_final if i == mlp_layers-1 else dropout_hidden

                mlp_modules.append( torch.nn.Linear(in_curr, out_curr, bias=bias, device=device, dtype=dtype) )

                if bn_curr:
                    mlp_modules.append( torch.nn.BatchNorm1d(num_features=out_curr, device=device, dtype=dtype) ) 

                if act_curr is not None:
                    mlp_modules.append( act_curr )

                if dropout_curr > 0:
                    mlp_modules.append( torch.nn.Dropout(p=dropout_curr) )
                
            self.mlp = torch.nn.Sequential(*mlp_modules); 
        
        self.size_coeff = torch.nn.Parameter( torch.ones(1, device=device, dtype=dtype) / np.sqrt(embed_dim-1), requires_grad=learnable_embedding)

        self.sw_embed = SW_embedding(d=in_channels, m=embed_dim-1, learnable_slices=learnable_embedding,
                                     learnable_freqs=learnable_embedding, 
                                     minimize_slice_coherence=True, freqs_init='spread',
                                     enable_bias=embed_bias,
                                     device=device, dtype=dtype)

        device = device if device is not None else self.sw_embed.get_device()
        dtype = dtype if dtype is not None else self.sw_embed.get_dtype()
        self.to(device=device, dtype=dtype)


    def forward(self, vertex_features, edge_index):
        # vertex_features has shape [num_vertices, in_channels]
        # edge_index has shape [2, num_edges]

        # Verify input
        assert vertex_features.dtype == self.sw_embed.get_dtype(), 'vertex_features has incorrect dtype (expected %s, got %s)' % (self.sw_embed.get_dtype(), vertex_features.dtype)
        assert vertex_features.device == self.sw_embed.get_device(), 'vertex_features has incorrect device (expected %s, got %s)' % (self.sw_embed.get_device(), vertex_features.device)
        assert edge_index.device == self.sw_embed.get_device(), 'edge_index has incorrect device (expected %s, got %s)' % (self.sw_embed.get_device(), edge_index.device)
        
        n = vertex_features.size(0)

        # This adds self-loops the old-fashioned way
        #edge_index, _ = add_self_loops(edge_index, num_nodes=n)

        # Calculate vertex degrees the old-fashioned way
        # row, col = edge_index
        # vertex_degrees = degree(col, n, dtype=vertex_features.dtype).unsqueeze(-1)

        # Convert edge_index to sparse adjacency matrix
        adj = pyg.utils.to_torch_coo_tensor(edge_index, edge_attr=None, size=n, is_coalesced=False)
        adj = adj.to(vertex_features.dtype)
        adj.requires_grad_(False)

        # Add weighted self-loops
        if self.self_loop_weight > 0:
            indices = torch.arange(n, dtype=torch.long, device=vertex_features.device).unsqueeze(0).expand(2, -1)
            values = torch.full((n,), self.self_loop_weight, dtype=vertex_features.dtype, device=vertex_features.device)
            adj += torch.sparse_coo_tensor(indices, values, (n, n))
            adj.coalesce()
            in_degrees = torch.sum(adj, dim=-1, keepdim=True).to_dense()
        else:
            adj.coalesce()

        # Aggregate neighboring vertex features
        emb = self.sw_embed(X=vertex_features, W=adj, graph_mode=True, serialize_num_slices=None)

        # Add neighborhood sizes multiplied by the norms of the neighborhood embeddings, and optionally also the self feature of each vertex
        emb_cat_list = (vertex_features,) if self.concat_self else ()
        emb_cat_list = emb_cat_list + (emb, (self.size_coeff*in_degrees*emb.norm(dim=-1,keepdim=True)))
        emb = torch.cat(emb_cat_list, dim=-1)

        # Apply MLP or dimensionality reduction to neighborhood embeddings
        if self.mlp is not None:
            out = self.mlp(emb)
        elif self.concat_self:
            out = torch.matmul(emb, self.dim_reduct.transpose(0,1))
        else:
            out = emb

        return out


    def aggregate(self):
        pass

    def message(self):
        pass

    def update(self):
        pass

