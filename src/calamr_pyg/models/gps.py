"""
GPS (General, Powerful, Scalable) Graph Transformer model.

GPS combines local message passing (GINEConv) with global attention,
representing the state-of-the-art for graph classification tasks.
Reference: Rampášek et al. "Recipe for a General, Powerful, Scalable Graph Transformer" (NeurIPS 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_batch


class GPS(nn.Module):
    """
    GPS (General, Powerful, Scalable) Graph Transformer.

    Key features:
    - Combines local MPNN (GINEConv) with global multi-head attention
    - GINEConv handles edge features natively
    - Global attention captures long-range dependencies
    - Designed for graph classification with state-of-the-art performance

    Args:
        input_dim: Node feature dimension (default: 771)
        hidden_dim: Hidden layer dimension (default: 256)
        num_layers: Number of GPS layers (default: 4)
        edge_dim: Edge feature dimension (default: 4)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.2)
        pooling: Graph pooling method ('mean', 'add', 'both')
    """

    def __init__(
        self,
        input_dim: int = 771,
        hidden_dim: int = 256,
        num_layers: int = 4,
        edge_dim: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        pooling: str = "mean",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # Input projection for nodes
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
        )

        # GPS layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # Local MPNN: GINEConv with edge features
            gin_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            local_conv = GINEConv(gin_nn, train_eps=True, edge_dim=hidden_dim)

            # GPS layer combines local MPNN + global attention
            gps_layer = GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=num_heads,
                dropout=dropout,
                norm="layer",  # LayerNorm for stability
            )
            self.convs.append(gps_layer)

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

        # Get edge features
        edge_attr = getattr(data, "edge_attr", None)

        # Encode node features
        x = self.node_encoder(x)

        # Encode edge features
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            # Create learnable edge features if not available
            edge_attr = torch.zeros(
                edge_index.size(1), x.size(1), device=x.device, dtype=x.dtype
            )

        # Apply GPS layers
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

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
