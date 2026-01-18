# Calamr-PyG

Medical hallucination detection using Graph Neural Networks with PyTorch Geometric.

## Project Structure

```
calamr_pyg/
├── src/calamr_pyg/          # Main package
│   ├── models/              # Model architectures
│   │   └── hybrid_gcn.py    # Hybrid GCN with attention pooling
│   ├── data/                # Data utilities
│   │   ├── dataset_utils.py # Dataset loading and splitting
│   │   └── datasets/        # Dataset implementations
│   ├── training/            # Training utilities
│   │   ├── trainer.py       # Training and evaluation loops
│   │   └── metrics.py       # Metrics and threshold optimization
│   └── utils/               # General utilities
├── scripts/                 # Entry point scripts
│   └── train_hybrid.py      # Training script
├── tests/                   # Unit tests
├── configs/                 # Configuration files
└── results/                 # Training results (gitignored)
```

## Installation

### Using uv (recommended)

```bash
# Install in development mode
uv pip install -e .

# With development dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Install in development mode
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Usage

### Training

Train a Hybrid GCN model:

```bash
# Example with real data path
uv run python scripts/train_hybrid.py \
    --dataset_path /Users/pranavherur/Documents/side_projects/calamr/calamr-master/medhallu/v8/labeled \
    --hidden_dim 256 \
    --num_layers 3 \
    --dropout 0.2 \
    --pooling attention \
    --multi_head \
    --num_heads 4 \
    --lr 0.001 \
    --epochs 50 \
    --batch_size 32 \
    --output_dir results

# Or simplified
uv run python scripts/train_hybrid.py --dataset_path /path/to/dataset
```

### Key Arguments

- `--dataset_path`: Path to directory containing `.pt` graph data files
- `--hidden_dim`: Hidden dimension size (default: 256)
- `--num_layers`: Number of GCN layers (default: 3)
- `--dropout`: Dropout rate (default: 0.2)
- `--pooling`: Pooling method - `attention`, `mean`, or `max` (default: attention)
- `--multi_head`: Enable multi-head attention pooling
- `--num_heads`: Number of attention heads if multi-head enabled (default: 4)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--patience`: Early stopping patience (default: 10)

### Using as a Library

```python
from calamr_pyg.models import HybridGCN
from calamr_pyg.data import load_dataset, split_dataset_random
from calamr_pyg.training import train_epoch, evaluate, compute_metrics

# Load data
dataset = load_dataset(Path("/path/to/data"))
train_data, val_data, test_data = split_dataset_random(dataset, [0.7, 0.15, 0.15])

# Create model
model = HybridGCN(
    input_dim=768,
    hidden_dim=256,
    num_layers=3,
    dropout=0.2,
    pooling="attention",
)

# Train
train_epoch(model, train_loader, optimizer, criterion, device)

# Evaluate
loss, preds, labels = evaluate(model, val_loader, criterion, device)
metrics = compute_metrics(labels, preds, threshold=0.5)
```

## Model Architecture

The Hybrid GCN combines:
- **GCN layers**: Graph Convolutional Networks for node feature propagation
- **Attention pooling**: Learnable attention mechanism for graph-level representation
- **Multi-head attention**: Optional multi-head attention for richer representations

## Dataset Format

The model expects PyTorch Geometric `Data` objects with:
- `x`: Node features (shape: [num_nodes, feature_dim])
- `edge_index`: Graph connectivity (shape: [2, num_edges])
- `y`: Binary label (0 or 1) for hallucination detection
- `batch`: Batch assignment for nodes (automatically added by DataLoader)

## Results

Training results are saved to the output directory with:
- `best_hybrid_model.pt`: Best model checkpoint
- `results.json`: Training metrics and configuration

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
# Format with black
black src/ scripts/ tests/

# Lint with ruff
ruff check src/ scripts/ tests/
```

## License

TBD
