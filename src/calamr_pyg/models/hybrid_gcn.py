#!/usr/bin/env python3
"""
Shared model definitions for Hybrid GCN + Attention Pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


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
        attention_scores = self.attention_net(x)  # [num_nodes, 1]
        
        # Apply attention within each graph
        graph_embeddings = []
        for i in range(batch.max() + 1):
            mask = batch == i
            graph_x = x[mask]  # nodes in graph i
            graph_scores = attention_scores[mask]  # attention scores for graph i
            
            # Softmax attention weights within this graph
            graph_weights = torch.softmax(graph_scores, dim=0)
            
            # Weighted sum of node features
            graph_embedding = torch.sum(graph_x * graph_weights, dim=0)
            graph_embeddings.append(graph_embedding)
        
        return torch.stack(graph_embeddings)


class HybridGCN(torch.nn.Module):
    """GCN + Attention Pooling for alignment-based hallucination detection."""

    def __init__(
        self,
        input_dim=770,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,
        pooling="attention",
        use_multi_head_attention=False,
        num_attention_heads=4,
        attention_activation="tanh",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.use_multi_head_attention = use_multi_head_attention
        
        # GCN layers (proven to work)
        self.gcn_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.gcn_layers.append(GCNConv(in_dim, hidden_dim))
        
        # Pooling layers
        if pooling == "attention":
            if use_multi_head_attention:
                # Multi-head attention pooling
                self.attention_heads = nn.ModuleList([
                    AttentionPooling(hidden_dim, attention_activation) 
                    for _ in range(num_attention_heads)
                ])
                classifier_input_dim = hidden_dim * num_attention_heads
            else:
                # Single attention head
                self.attention_pool = AttentionPooling(hidden_dim, attention_activation)
                classifier_input_dim = hidden_dim
        else:
            classifier_input_dim = hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.dropout = dropout
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GCN layers
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == "attention":
            if self.use_multi_head_attention:
                # Multi-head attention pooling
                head_outputs = []
                for attention_head in self.attention_heads:
                    head_output = attention_head(x, batch)
                    head_outputs.append(head_output)
                x = torch.cat(head_outputs, dim=1)  # concatenate heads
            else:
                # Single attention head
                x = self.attention_pool(x, batch)
        elif self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Classification
        x = self.classifier(x)
        return x.squeeze(-1)