"""Model implementations for hallucination detection."""

from calamr_pyg.models.hybrid_gcn import HybridGCN
from calamr_pyg.models.edge_aware_gat import EdgeAwareGAT
from calamr_pyg.models.graph_sage import GraphSAGE
from calamr_pyg.models.gin import GIN
from calamr_pyg.models.gated_gnn import GatedGNN
from calamr_pyg.models.graph_transformer import GraphTransformer
from calamr_pyg.models.gps import GPS
from calamr_pyg.models.pna import PNA, compute_degree_histogram
from calamr_pyg.models.gatv2 import GATv2
from calamr_pyg.models.appnp import APPNP
from calamr_pyg.models.gcn2 import GCN2
from calamr_pyg.models.deeper_gcn import DeeperGCN
from calamr_pyg.models.graph_saint import GraphSAINT

__all__ = [
    "HybridGCN",
    "EdgeAwareGAT",
    "GraphSAGE",
    "GIN",
    "GatedGNN",
    "GraphTransformer",
    "GPS",
    "PNA",
    "compute_degree_histogram",
    "GATv2",
    "APPNP",
    "GCN2",
    "DeeperGCN",
    "GraphSAINT",
]
