# Simple working example of FSW Embedding

import numpy as np

import torch
import torch.nn as nn

from fsw_embedding import FSW_embedding

device = 'cuda'
dtype = torch.float32

batch_dims = (3,2,5)

d = 20
n = 100
embed_dim = 1000

emb = FSW_embedding(d_in = d, d_out=embed_dim, device=device, dtype=dtype)

# Create a batch of point clouds, each containing n points in R^d
X = torch.randn( batch_dims + (n,d), device=device, dtype=dtype)

# Create weights for the points
W = torch.softmax(torch.randn( batch_dims + (n, ), device=device, dtype=dtype), dim=-1)

print('')
print('Feature dimension: %d  Size of each multiset: %d  Embedding dimension: %d  Batch dimensions: %s' % (d,n,embed_dim,batch_dims))

X_emb = emb(X, W) # Embed ignoring weights
#X_emb = emb(X, W) # Embed with weights

print('Size of X: ', X.shape)
print('Size of E(X): ', X_emb.shape)

