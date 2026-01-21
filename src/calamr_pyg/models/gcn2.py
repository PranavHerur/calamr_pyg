#!/usr/bin/env python3
"""
GCN2 (GCNII) - GCN with Initial residual and Identity mapping.
Allows very deep GCNs without over-smoothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv, global_mean_pool, global_add_pool


class GCN2(nn.Module):
    """
    GCNII for graph classification.

    GCNII adds two techniques to enable deep GCNs:
    1. Initial residual connection: connects to initial representation
    2. Identity mapping: adds identity matrix to weight matrix

    Reference: Chen et al. "Simple and Deep Graph Convolutional Networks" (ICML 2020)

    Args:
        input_dim: Node feature dimension (default: 771)
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of GCNII layers (default: 8)
        alpha: Initial residual weight (default: 0.1)
        theta: Identity mapping weight (default: 0.5)
        dropout: Dropout rate (default: 0.2)
        shared_weights: Share weights across layers (default: False)
        pooling: Graph pooling - 'mean', 'add', 'both' (default: 'mean')
    """

    def __init__(
        self,
        input_dim=771,
        hidden_dim=256,
        num_layers=8,
        alpha=0.1,
        theta=0.5,
        dropout=0.2,
        shared_weights=False,
        pooling="mean",
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # Initial projection to hidden dimension
        self.lin_in = nn.Linear(input_dim, hidden_dim)

        # GCNII layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(
                GCN2Conv(
                    hidden_dim,
                    alpha=alpha,
                    theta=theta,
                    layer=i + 1,
                    shared_weights=shared_weights,
                    normalize=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

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

        # Classifier
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

        # Initial projection
        x = self.lin_in(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Store initial representation for residual connections
        x_0 = x

        # GCNII layers with initial residual
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, x_0, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling
        x = self.pool(x, batch)

        # Classification
        return self.classifier(x).squeeze(-1)
