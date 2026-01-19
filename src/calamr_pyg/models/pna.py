"""
PNA (Principal Neighborhood Aggregation) model.

PNA uses multiple aggregators (mean, max, min, std) and scalers to create
expressive graph representations. It's particularly effective when
different aggregation strategies capture different aspects of the data.
Reference: Corso et al. "Principal Neighbourhood Aggregation for Graph Nets" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, global_mean_pool, global_add_pool, BatchNorm
from torch_geometric.utils import degree


class PNA(nn.Module):
    """
    PNA (Principal Neighborhood Aggregation) model.

    Key features:
    - Multiple aggregators: mean, max, min, std
    - Multiple scalers: identity, amplification, attenuation
    - Edge feature support
    - Captures diverse neighborhood patterns

    Args:
        input_dim: Node feature dimension (default: 771)
        hidden_dim: Hidden layer dimension (default: 256)
        num_layers: Number of PNA layers (default: 4)
        edge_dim: Edge feature dimension (default: 4)
        dropout: Dropout probability (default: 0.2)
        pooling: Graph pooling method ('mean', 'add', 'both')
        deg: Degree histogram for PNA scalers (computed from data if None)
    """

    def __init__(
        self,
        input_dim: int = 771,
        hidden_dim: int = 256,
        num_layers: int = 4,
        edge_dim: int = 4,
        dropout: float = 0.2,
        pooling: str = "both",
        deg: torch.Tensor = None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # PNA aggregators and scalers
        aggregators = ["mean", "max", "min", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        # Default degree tensor if not provided
        # This should be computed from the actual dataset for best results
        if deg is None:
            # Default: assume max degree of 50 with uniform distribution
            deg = torch.ones(51)

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Edge encoder
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # PNA layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            conv = PNAConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=hidden_dim,
                towers=4,  # Number of towers for PNA
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            self.convs.append(conv)
            self.norms.append(BatchNorm(hidden_dim))

        # Graph-level pooling
        if pooling == "both":
            pool_dim = hidden_dim * 2
        else:
            pool_dim = hidden_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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

        # Project node features
        x = self.input_proj(x)

        # Encode edge features
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = torch.zeros(
                edge_index.size(1), x.size(1), device=x.device, dtype=x.dtype
            )

        # Apply PNA layers
        for conv, norm in zip(self.convs, self.norms):
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


def compute_degree_histogram(dataset, max_degree=100):
    """
    Compute degree histogram from a dataset for PNA.

    Args:
        dataset: PyG dataset or list of Data objects
        max_degree: Maximum degree to consider

    Returns:
        Degree tensor for PNA initialization
    """
    deg = torch.zeros(max_degree + 1, dtype=torch.long)

    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        d = d.clamp(max=max_degree)
        deg += torch.bincount(d, minlength=max_degree + 1)

    return deg
