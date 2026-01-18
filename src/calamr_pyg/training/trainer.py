"""Training and evaluation functions."""

import torch
import torch.nn as nn
import numpy as np


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Tuple of (average_loss, predictions, labels)
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    debug_first_batch = True

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss = criterion(out, batch.y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds = torch.sigmoid(out)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

            if debug_first_batch:
                print(
                    f"  Raw outputs: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}"
                )
                print(
                    f"  Predictions: min={preds.min().item():.4f}, max={preds.max().item():.4f}, mean={preds.mean().item():.4f}"
                )
                debug_first_batch = False

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on validation/test data.

    Args:
        model: PyTorch model
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average_loss, predictions, labels)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.float())
            total_loss += loss.item()

            preds = torch.sigmoid(out)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)
