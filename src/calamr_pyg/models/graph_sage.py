#!/usr/bin/env python3
"""
GraphSAGE for hallucination detection.
Sampling-based aggregation, good for inductive learning and variable graph sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_add_pool


class AttentionPooling(nn.Module):
    """Attention-based global pooling."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, batch):
        scores = self.attention(x)
        embeddings = []
        for i in range(batch.max() + 1):
            mask = batch == i
            weights = torch.softmax(scores[mask], dim=0)
            embeddings.append((x[mask] * weights).sum(dim=0))
        return torch.stack(embeddings)


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for graph classification.

    Uses sampling and aggregation approach - learns to aggregate
    feature information from a node's local neighborhood.

    Reference: Hamilton et al. "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)

    Args:
        input_dim: Node feature dimension (default: 771)
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of SAGE layers (default: 3)
        dropout: Dropout rate (default: 0.2)
        aggregator: Aggregation method - 'mean', 'max', 'lstm' (default: 'mean')
        pooling: Graph pooling - 'mean', 'add', 'attention' (default: 'attention')
    """

    def __init__(
        self,
        input_dim=771,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,
        aggregator="mean",
        pooling="attention",
    ):
        super().__init__()
        self.dropout = dropout
        self.pooling_type = pooling

        # SAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggregator))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Pooling
        if pooling == "attention":
            self.pool = AttentionPooling(hidden_dim)
        elif pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "add":
            self.pool = global_add_pool

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
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

        # Message passing
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling
        if self.pooling_type == "attention":
            x = self.pool(x, batch)
        else:
            x = self.pool(x, batch)

        # Classification
        return self.classifier(x).squeeze(-1)
