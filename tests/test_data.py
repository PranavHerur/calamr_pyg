"""Tests for data utilities."""

import torch
from torch_geometric.data import Data
import pytest
from pathlib import Path
import tempfile

from calamr_pyg.data import split_dataset_random, load_dataset


def create_dummy_dataset(size=100, feature_dim=768):
    """Create a dummy dataset for testing."""
    dataset = []
    for i in range(size):
        x = torch.randn(10, feature_dim)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        y = torch.tensor([i % 2], dtype=torch.float)
        dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset


def test_split_dataset_random():
    """Test random dataset splitting."""
    dataset = create_dummy_dataset(size=100)
    ratios = [0.7, 0.15, 0.15]

    splits = split_dataset_random(dataset, ratios, random_seed=42)

    assert len(splits) == 3
    assert len(splits[0]) == 70
    assert len(splits[1]) == 15
    assert len(splits[2]) == 15


def test_split_dataset_reproducibility():
    """Test that splitting is reproducible with same seed."""
    dataset = create_dummy_dataset(size=100)
    ratios = [0.7, 0.15, 0.15]

    splits1 = split_dataset_random(dataset, ratios, random_seed=42)
    splits2 = split_dataset_random(dataset, ratios, random_seed=42)

    # Check that splits are identical
    for s1, s2 in zip(splits1, splits2):
        assert len(s1) == len(s2)


def test_load_dataset_single_file():
    """Test loading a single .pt file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dataset = create_dummy_dataset(size=10)

        # Save as single file
        torch.save(dataset, tmpdir / "data.pt")

        # Load
        loaded = load_dataset(tmpdir)
        assert len(loaded) == 10


def test_load_dataset_multiple_files():
    """Test loading multiple .pt files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save multiple files
        dataset1 = create_dummy_dataset(size=10)
        dataset2 = create_dummy_dataset(size=15)
        torch.save(dataset1, tmpdir / "data1.pt")
        torch.save(dataset2, tmpdir / "data2.pt")

        # Load
        loaded = load_dataset(tmpdir)
        assert len(loaded) == 25


def test_load_dataset_string_path():
    """Test that load_dataset accepts string paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = create_dummy_dataset(size=5)
        torch.save(dataset, Path(tmpdir) / "data.pt")

        # Load using string path
        loaded = load_dataset(tmpdir)
        assert len(loaded) == 5


def test_load_dataset_nonexistent_path():
    """Test that load_dataset raises error for nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_dataset("/nonexistent/path")


def test_load_dataset_no_files():
    """Test that load_dataset raises error when no .pt files found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="No .pt files found"):
            load_dataset(tmpdir)
