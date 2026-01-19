#!/usr/bin/env python3
"""
Gated Graph Neural Network for hallucination detection.
Uses GRU-style gating for message passing - good for sequential/text data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_add_pool


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


class GatedGNN(nn.Module):
    """
    Gated Graph Neural Network for graph classification.

    Uses GRU-style gating mechanism for message passing.
    Good for capturing sequential dependencies in text-based graphs.

    Reference: Li et al. "Gated Graph Sequence Neural Networks" (ICLR 2016)

    Args:
        input_dim: Node feature dimension (default: 771)
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of message passing steps (default: 5)
        dropout: Dropout rate (default: 0.2)
        pooling: Graph pooling - 'mean', 'add', 'attention' (default: 'attention')
    """

    def __init__(
        self,
        input_dim=771,
        hidden_dim=256,
        num_layers=5,
        dropout=0.2,
        pooling="attention",
    ):
        super().__init__()
        self.dropout = dropout
        self.pooling_type = pooling

        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Gated Graph Conv - note: out_channels must equal in_channels
        # num_layers here means number of message passing iterations
        self.conv = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)

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

        # Project to hidden dimension
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Gated message passing
        x = self.conv(x, edge_index)

        # Pooling
        if self.pooling_type == "attention":
            x = self.pool(x, batch)
        else:
            x = self.pool(x, batch)

        # Classification
        return self.classifier(x).squeeze(-1)
