"""Tests for model architectures."""

import torch
from torch_geometric.data import Data, Batch
import pytest

from calamr_pyg.models import HybridGCN


def create_sample_graph(num_nodes=10, feature_dim=768):
    """Create a sample graph for testing."""
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    y = torch.tensor([1.0])
    return Data(x=x, edge_index=edge_index, y=y)


def test_hybrid_gcn_forward():
    """Test HybridGCN forward pass."""
    model = HybridGCN(
        input_dim=768,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        pooling="attention",
    )

    # Create sample data
    data = create_sample_graph(num_nodes=10, feature_dim=768)
    data.batch = torch.zeros(10, dtype=torch.long)

    # Forward pass
    output = model(data)

    # Check output shape
    assert output.shape == torch.Size([1])
    assert not torch.isnan(output).any()


def test_hybrid_gcn_pooling_methods():
    """Test different pooling methods."""
    for pooling in ["attention", "mean", "max"]:
        model = HybridGCN(
            input_dim=768,
            hidden_dim=128,
            num_layers=2,
            pooling=pooling,
        )

        data = create_sample_graph(num_nodes=10, feature_dim=768)
        data.batch = torch.zeros(10, dtype=torch.long)

        output = model(data)
        assert output.shape == torch.Size([1])


def test_hybrid_gcn_multi_head():
    """Test multi-head attention pooling."""
    model = HybridGCN(
        input_dim=768,
        hidden_dim=128,
        num_layers=2,
        pooling="attention",
        use_multi_head_attention=True,
        num_attention_heads=4,
    )

    data = create_sample_graph(num_nodes=10, feature_dim=768)
    data.batch = torch.zeros(10, dtype=torch.long)

    output = model(data)
    assert output.shape == torch.Size([1])


def test_hybrid_gcn_batch():
    """Test model with batched data."""
    model = HybridGCN(
        input_dim=768,
        hidden_dim=128,
        num_layers=2,
        pooling="attention",
    )

    # Create batch of graphs
    graphs = [create_sample_graph(num_nodes=10, feature_dim=768) for _ in range(4)]
    batch = Batch.from_data_list(graphs)

    output = model(batch)
    assert output.shape == torch.Size([4])
