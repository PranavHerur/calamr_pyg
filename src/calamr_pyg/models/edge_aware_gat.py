#!/usr/bin/env python3
"""
Edge-Aware Graph Attention Network for hallucination detection.
Uses edge features (flow values) to better capture alignment strength.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class AttentionPooling(nn.Module):
    """Attention-based global pooling for graph-level representation."""

    def __init__(self, hidden_dim, activation="tanh"):
        super().__init__()
        activation_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation_fn,
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, batch):
        # Compute attention weights
        attention_scores = self.attention_net(x)

        # Apply attention within each graph
        graph_embeddings = []
        for i in range(batch.max() + 1):
            mask = batch == i
            graph_x = x[mask]
            graph_scores = attention_scores[mask]

            # Softmax attention weights within this graph
            graph_weights = torch.softmax(graph_scores, dim=0)

            # Weighted sum of node features
            graph_embedding = torch.sum(graph_x * graph_weights, dim=0)
            graph_embeddings.append(graph_embedding)

        return torch.stack(graph_embeddings)


class EdgeAwareGAT(nn.Module):
    """
    Graph Attention Network that uses edge features.

    This model improves upon HybridGCN by using GATConv layers that can
    incorporate edge attributes (specifically the flow values) to better
    capture alignment strength between source and summary.

    Args:
        input_dim: Node feature dimension (default: 771)
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of GAT layers (default: 3)
        edge_dim: Edge feature dimension (default: 4)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.2)
        pooling: Pooling method - 'attention', 'mean', 'max' (default: 'attention')
        attention_activation: Activation for attention pooling (default: 'tanh')
    """

    def __init__(
        self,
        input_dim=771,
        hidden_dim=256,
        num_layers=3,
        edge_dim=4,
        num_heads=4,
        dropout=0.2,
        pooling="attention",
        attention_activation="tanh",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.num_heads = num_heads
        self.dropout = dropout

        # GAT layers with edge features
        self.gat_layers = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        self.gat_layers.append(
            GATConv(
                input_dim,
                hidden_dim // num_heads,  # Output per head
                heads=num_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                concat=True,  # Concatenate heads
            )
        )

        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=True,
                )
            )

        # Last layer: hidden_dim -> hidden_dim (average heads)
        self.gat_layers.append(
            GATConv(
                hidden_dim,
                hidden_dim,
                heads=num_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                concat=False,  # Average heads for final layer
            )
        )

        # Pooling layer
        if pooling == "attention":
            self.pool = AttentionPooling(hidden_dim, attention_activation)
        elif pooling == "mean":
            self.pool = global_mean_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Data object with:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_dim]
                - batch: Batch assignment [num_nodes]

        Returns:
            Logits for binary classification [batch_size]
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Apply GAT layers with edge features
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index, edge_attr=edge_attr)

            # Apply activation and dropout (except last layer gets handled differently)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer activation
        x = F.elu(x)

        # Global pooling
        if self.pooling == "attention":
            x = self.pool(x, batch)
        else:
            x = self.pool(x, batch)

        # Classification
        x = self.classifier(x)
        return x.squeeze(-1)

    def get_attention_weights(self, data):
        """
        Get attention weights from GAT layers for visualization.

        Returns:
            List of attention weight tensors for each layer.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        attention_weights = []

        for gat_layer in self.gat_layers:
            x, (edge_index_out, alpha) = gat_layer(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            attention_weights.append(alpha)
            x = F.elu(x)

        return attention_weights
