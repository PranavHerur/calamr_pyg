#!/usr/bin/env python3
"""
DeeperGCN - Deep Graph Convolutional Network using GENConv.
Uses generalized message aggregation with skip connections for very deep networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv, DeepGCNLayer, global_mean_pool, global_add_pool


class DeeperGCN(nn.Module):
    """
    DeeperGCN for graph classification.

    Uses GENConv (Generalized Aggregation) with DeepGCNLayer that includes
    pre-activation residual connections, allowing networks of 100+ layers.

    Reference: Li et al. "DeeperGCN: All You Need to Train Deeper GCNs" (2020)

    Args:
        input_dim: Node feature dimension (default: 771)
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of DeepGCN layers (default: 7)
        dropout: Dropout rate (default: 0.2)
        aggr: Aggregation method - 'softmax', 'powermean', 'add', 'mean', 'max' (default: 'softmax')
        t: Temperature for softmax aggregation (default: 1.0)
        p: Power for powermean aggregation (default: 1.0)
        learn_t: Learn temperature parameter (default: True)
        learn_p: Learn power parameter (default: True)
        pooling: Graph pooling - 'mean', 'add', 'both' (default: 'mean')
    """

    def __init__(
        self,
        input_dim=771,
        hidden_dim=256,
        num_layers=7,
        dropout=0.2,
        aggr="softmax",
        t=1.0,
        p=1.0,
        learn_t=True,
        learn_p=True,
        pooling="mean",
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # Initial projection
        self.lin_in = nn.Linear(input_dim, hidden_dim)

        # DeepGCN layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            conv = GENConv(
                hidden_dim,
                hidden_dim,
                aggr=aggr,
                t=t,
                p=p,
                learn_t=learn_t,
                learn_p=learn_p,
                num_layers=2,  # MLP layers inside GENConv
                norm="layer",
            )
            norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(
                conv,
                norm,
                act,
                block="res+",  # Pre-activation residual
                dropout=dropout,
            )
            self.layers.append(layer)

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

        # DeepGCN layers
        for layer in self.layers:
            x = layer(x, edge_index)

        # Final activation (DeepGCNLayer skips it at the end)
        x = self.layers[0].act(self.layers[0].norm(x))

        # Graph-level pooling
        x = self.pool(x, batch)

        # Classification
        return self.classifier(x).squeeze(-1)
