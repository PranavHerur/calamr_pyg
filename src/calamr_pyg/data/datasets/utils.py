"""Dataset splitting and loading utilities."""

from typing import List, Sequence, Union, cast
from pathlib import Path
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data

import logging
logger = logging.getLogger(__name__)


def split_dataset_random(
    dataset: Union[Dataset[Data], Sequence[Data]],
    ratios: List[float],
    random_seed: int = 42,
) -> Sequence[Subset[Data]]:
    """
    Split dataset into train/val/test sets using random split.

    Args:
        dataset: Dataset or sequence of PyG Data objects
        ratios: List of split ratios (e.g., [0.7, 0.15, 0.15])
        random_seed: Random seed for reproducibility

    Returns:
        List of dataset splits
    """
    gen = torch.Generator().manual_seed(random_seed)
    # Cast to Dataset for type checking - sequences work at runtime
    # since they implement __len__ and __getitem__
    return torch.utils.data.random_split(cast(Dataset[Data], dataset), ratios, generator=gen)


def load_dataset(path: Union[str, Path]) -> List[Data]:
    """
    Load a dataset from a directory path containing PyTorch .pt files.

    This function supports multiple scenarios:
    1. Single .pt file in directory - loads that file
    2. Multiple .pt files in directory - loads all files and concatenates them
    3. Nested directories - searches recursively for .pt files

    Args:
        path: Path to directory containing .pt files (can be string or Path object)
              Example: "/Users/name/data/medhallu/v8/labeled"

    Returns:
        List of loaded PyG Data objects

    Raises:
        ValueError: If no data files found
        FileNotFoundError: If path doesn't exist

    Example:
        >>> dataset = load_dataset("/path/to/medhallu/v8/labeled")
        >>> print(f"Loaded {len(dataset)} graphs")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if not path.is_dir():
        raise ValueError(f"Path must be a directory: {path}")

    # Find all .pt files in the directory
    data_files = sorted(list(path.glob("*.pt")))

    # If no files in immediate directory, try recursive search
    if len(data_files) == 0:
        data_files = sorted(list(path.rglob("*.pt")))

    if len(data_files) == 0:
        raise ValueError(f"No .pt files found in {path}")

    # Load all data files
    all_data = []
    for file in data_files:
        logger.debug(f"Loading {file.name}...")
        data = torch.load(file, weights_only=False)

        # Handle different formats
        if isinstance(data, list):
            all_data.extend(data)
        elif isinstance(data, Data):
            all_data.append(data)
        else:
            raise ValueError(f"Unexpected data type in {file}: {type(data)}")

    logger.info(f"Loaded {len(all_data)} graphs from {len(data_files)} file(s)")
    return all_data
