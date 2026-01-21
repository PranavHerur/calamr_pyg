#!/usr/bin/env python3
"""
GATv2 - Improved Graph Attention Network for hallucination detection.
Fixes the "static attention" problem in the original GAT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool


class GATv2(nn.Module):
    """
    GATv2 for graph classification.

    GATv2 computes dynamic attention that depends on both the query and key nodes,
    unlike GAT which computes static attention (same for all queries).

    Reference: Brody et al. "How Attentive are Graph Attention Networks?" (ICLR 2022)

    Args:
        input_dim: Node feature dimension (default: 771)
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of GATv2 layers (default: 3)
        heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.2)
        pooling: Graph pooling - 'mean', 'add', 'both' (default: 'mean')
    """

    def __init__(
        self,
        input_dim=771,
        hidden_dim=256,
        num_layers=3,
        heads=4,
        dropout=0.2,
        pooling="mean",
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # GATv2 layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim * heads
            # Last layer uses 1 head for final representation
            out_heads = 1 if i == num_layers - 1 else heads
            concat = i < num_layers - 1

            self.convs.append(
                GATv2Conv(
                    in_dim,
                    hidden_dim,
                    heads=out_heads,
                    concat=concat,
                    dropout=dropout,
                    add_self_loops=True,
                    share_weights=False,
                )
            )
            out_dim = hidden_dim * out_heads if concat else hidden_dim
            self.norms.append(nn.LayerNorm(out_dim))

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

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling
        x = self.pool(x, batch)

        # Classification
        return self.classifier(x).squeeze(-1)
