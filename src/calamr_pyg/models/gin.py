#!/usr/bin/env python3
"""
Graph Isomorphism Network (GIN) for hallucination detection.
Most expressive GNN - can distinguish different graph structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class GIN(nn.Module):
    """
    Graph Isomorphism Network for graph classification.

    GIN is provably the most expressive GNN under the WL test framework.
    Uses MLPs for neighbor aggregation instead of simple mean/sum.

    Reference: Xu et al. "How Powerful are Graph Neural Networks?" (ICLR 2019)

    Args:
        input_dim: Node feature dimension (default: 771)
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of GIN layers (default: 3)
        dropout: Dropout rate (default: 0.2)
        train_eps: Whether to learn epsilon (default: True)
        pooling: Graph pooling - 'add', 'mean' (default: 'add')
    """

    def __init__(
        self,
        input_dim=771,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,
        train_eps=True,
        pooling="add",
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # GIN layers with MLPs
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim

            # MLP for each GIN layer
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Pooling
        self.pool = global_add_pool if pooling == "add" else global_mean_pool

        # Jumping knowledge - concatenate all layer outputs
        self.use_jk = True
        jk_dim = hidden_dim * num_layers if self.use_jk else hidden_dim

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
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

        # Collect outputs from each layer for jumping knowledge
        layer_outputs = []

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

        # Jumping knowledge - concatenate all layer representations
        if self.use_jk:
            # Pool each layer's output and concatenate
            pooled = [self.pool(h, batch) for h in layer_outputs]
            x = torch.cat(pooled, dim=-1)
        else:
            x = self.pool(x, batch)

        # Classification
        return self.classifier(x).squeeze(-1)
