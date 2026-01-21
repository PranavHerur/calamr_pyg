#!/usr/bin/env python3
"""
GraphSAINT - Graph Sampling Based Inductive Learning.
Efficient sampling-based training for large graphs.
Uses a simple but effective GCN backbone designed for sampled subgraphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_add_pool


class GraphSAINT(nn.Module):
    """
    GraphSAINT-style model for graph classification.

    Uses GraphSAGE convolutions which are naturally suited for sampled
    subgraphs since they sample neighbors during training. The model
    is designed to work well with mini-batch training on subgraph samples.

    Reference: Zeng et al. "GraphSAINT: Graph Sampling Based Inductive
    Learning Method" (ICLR 2020)

    Args:
        input_dim: Node feature dimension (default: 771)
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of SAGE layers (default: 3)
        dropout: Dropout rate (default: 0.2)
        normalize: Normalize output embeddings (default: True)
        pooling: Graph pooling - 'mean', 'add', 'both' (default: 'mean')
    """

    def __init__(
        self,
        input_dim=771,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,
        normalize=True,
        pooling="mean",
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.normalize = normalize

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.convs.append(
                SAGEConv(
                    in_dim,
                    hidden_dim,
                    normalize=normalize,
                    project=True,  # Project before aggregation
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Skip connection from input
        self.skip_lin = nn.Linear(input_dim, hidden_dim)

        # Pooling
        if pooling == "both":
            self.pool = lambda x, batch: torch.cat(
                [global_mean_pool(x, batch), global_add_pool(x, batch)], dim=-1
            )
            pool_dim = hidden_dim * 2
        elif pooling == "add":
            self.pool = global_add_pool
            pool_dim = hidden_dim
        else:
            self.pool = global_mean_pool
            pool_dim = hidden_dim

        # Classifier with residual
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Skip connection from input
        x_skip = self.skip_lin(x)

        # Message passing layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Add skip connection
        x = x + x_skip

        # Graph-level pooling
        x = self.pool(x, batch)

        # Classification
        return self.classifier(x).squeeze(-1)
