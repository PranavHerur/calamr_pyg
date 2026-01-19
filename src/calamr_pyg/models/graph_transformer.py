"""
Graph Transformer model using TransformerConv.

TransformerConv brings transformer-style attention to GNNs with native
edge feature support, making it ideal for graphs with rich edge attributes
like alignment graphs with flow information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_add_pool


class GraphTransformer(nn.Module):
    """
    Graph Transformer using TransformerConv layers.

    Key features:
    - Transformer-style multi-head attention for message passing
    - Native edge feature integration (critical for flow-based graphs)
    - Concatenates attention head outputs (like original Transformer)
    - Position-aware through edge features

    Args:
        input_dim: Node feature dimension (default: 771)
        hidden_dim: Hidden layer dimension (default: 256)
        num_layers: Number of TransformerConv layers (default: 4)
        edge_dim: Edge feature dimension (default: 4)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.2)
        pooling: Graph pooling method ('mean', 'add', 'both')
    """

    def __init__(
        self,
        input_dim: int = 771,
        hidden_dim: int = 256,
        num_layers: int = 4,
        edge_dim: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
        pooling: str = "both",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Edge feature projection
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # Transformer conv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # TransformerConv concatenates heads, so output is heads * out_channels
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim,  # Projected edge features
                    dropout=dropout,
                    concat=True,  # Concatenate attention heads
                    beta=True,  # Use beta attention (learnable skip)
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Graph-level pooling
        if pooling == "both":
            pool_dim = hidden_dim * 2
        else:
            pool_dim = hidden_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Get edge features if available
        edge_attr = getattr(data, "edge_attr", None)

        # Project node features
        x = self.input_proj(x)

        # Project edge features
        if edge_attr is not None:
            edge_attr = self.edge_proj(edge_attr)
        else:
            # Create zero edge features if not available
            edge_attr = torch.zeros(
                edge_index.size(1), x.size(1), device=x.device, dtype=x.dtype
            )

        # Apply transformer conv layers with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection

        # Graph-level pooling
        if self.pooling == "mean":
            graph_embedding = global_mean_pool(x, batch)
        elif self.pooling == "add":
            graph_embedding = global_add_pool(x, batch)
        else:  # both
            mean_pool = global_mean_pool(x, batch)
            add_pool = global_add_pool(x, batch)
            graph_embedding = torch.cat([mean_pool, add_pool], dim=-1)

        # Classification
        out = self.classifier(graph_embedding)
        return out.squeeze(-1)
