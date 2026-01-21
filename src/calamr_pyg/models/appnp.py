#!/usr/bin/env python3
"""
APPNP - Approximate Personalized Propagation of Neural Predictions.
Decouples feature transformation from propagation using personalized PageRank.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP as APPNPConv, global_mean_pool, global_add_pool


class APPNP(nn.Module):
    """
    APPNP for graph classification.

    APPNP separates the neural network from the propagation scheme, using
    personalized PageRank to propagate predictions. This allows capturing
    long-range dependencies efficiently.

    Reference: Klicpera et al. "Predict then Propagate: Graph Neural Networks
    meet Personalized PageRank" (ICLR 2019)

    Args:
        input_dim: Node feature dimension (default: 771)
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of MLP layers before propagation (default: 2)
        K: Number of propagation steps (default: 10)
        alpha: Teleport probability (default: 0.1)
        dropout: Dropout rate (default: 0.2)
        pooling: Graph pooling - 'mean', 'add', 'both' (default: 'mean')
    """

    def __init__(
        self,
        input_dim=771,
        hidden_dim=256,
        num_layers=2,
        K=10,
        alpha=0.1,
        dropout=0.2,
        pooling="mean",
    ):
        super().__init__()
        self.dropout = dropout

        # Feature transformation MLP (before propagation)
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*layers)

        # APPNP propagation layer
        self.prop = APPNPConv(K=K, alpha=alpha, dropout=dropout)

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

        # Transform features with MLP
        x = self.mlp(x)

        # Propagate with personalized PageRank
        x = self.prop(x, edge_index)

        # Graph-level pooling
        x = self.pool(x, batch)

        # Classification
        return self.classifier(x).squeeze(-1)
