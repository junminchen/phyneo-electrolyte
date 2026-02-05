"""
Sub-graph Neural Network for molecular representations.

This module implements a graph neural network that operates on molecular
sub-graphs to learn local chemical environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubgraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for learning molecular representations.
    
    This network processes molecular graphs using message passing and
    learns representations of local chemical environments.
    
    Args:
        node_input_dim (int): Dimension of input node features (default: 32)
        edge_input_dim (int): Dimension of input edge features (default: 16)
        hidden_dim (int): Dimension of hidden layers (default: 128)
        output_dim (int): Dimension of output features (default: 64)
        n_layers (int): Number of message passing layers (default: 3)
        aggregation (str): Aggregation method ('sum', 'mean', 'max') (default: 'sum')
        dropout_rate (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        node_input_dim=32,
        edge_input_dim=16,
        hidden_dim=128,
        output_dim=64,
        n_layers=3,
        aggregation="sum",
        dropout_rate=0.1
    ):
        super(SubgraphNeuralNetwork, self).__init__()
        
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.aggregation = aggregation
        
        # Node embedding layer
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        
        # Edge embedding layer
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim)
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            MessagePassingLayer(
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, node_features, edge_features, edge_index):
        """
        Forward pass through the graph neural network.
        
        Args:
            node_features (torch.Tensor): Node features, shape (n_nodes, node_input_dim)
            edge_features (torch.Tensor): Edge features, shape (n_edges, edge_input_dim)
            edge_index (torch.Tensor): Edge connectivity, shape (2, n_edges)
                                      edge_index[0] = source nodes
                                      edge_index[1] = target nodes
        
        Returns:
            torch.Tensor: Node representations, shape (n_nodes, output_dim)
        """
        # Embed nodes and edges
        node_h = self.node_embedding(node_features)
        edge_h = self.edge_embedding(edge_features)
        
        # Message passing
        for layer in self.message_layers:
            node_h = layer(node_h, edge_h, edge_index)
        
        # Output projection
        output = self.output_projection(node_h)
        
        return output
    
    def aggregate_graph(self, node_representations, batch_index=None):
        """
        Aggregate node representations to graph-level representation.
        
        Args:
            node_representations (torch.Tensor): Node features, shape (n_nodes, output_dim)
            batch_index (torch.Tensor): Batch index for each node, shape (n_nodes,)
                                       If None, assumes single graph
        
        Returns:
            torch.Tensor: Graph-level representation
        """
        if batch_index is None:
            # Single graph
            if self.aggregation == "sum":
                return node_representations.sum(dim=0, keepdim=True)
            elif self.aggregation == "mean":
                return node_representations.mean(dim=0, keepdim=True)
            elif self.aggregation == "max":
                return node_representations.max(dim=0, keepdim=True)[0]
        else:
            # Batched graphs
            batch_size = batch_index.max().item() + 1
            graph_representations = []
            
            for i in range(batch_size):
                mask = batch_index == i
                node_subset = node_representations[mask]
                
                if self.aggregation == "sum":
                    graph_rep = node_subset.sum(dim=0)
                elif self.aggregation == "mean":
                    graph_rep = node_subset.mean(dim=0)
                elif self.aggregation == "max":
                    graph_rep = node_subset.max(dim=0)[0]
                
                graph_representations.append(graph_rep)
            
            return torch.stack(graph_representations)
    
    def extra_repr(self):
        """String representation of the module."""
        return (
            f"node_input_dim={self.node_input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"output_dim={self.output_dim}, "
            f"n_layers={self.n_layers}"
        )


class MessagePassingLayer(nn.Module):
    """
    Single message passing layer for graph neural network.
    
    Args:
        hidden_dim (int): Dimension of hidden features
        dropout_rate (float): Dropout rate
    """
    
    def __init__(self, hidden_dim=128, dropout_rate=0.1):
        super(MessagePassingLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Message function
        self.message_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),  # [node, neighbor, edge]
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # [node, aggregated_messages]
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, node_h, edge_h, edge_index):
        """
        Perform message passing.
        
        Args:
            node_h (torch.Tensor): Node features, shape (n_nodes, hidden_dim)
            edge_h (torch.Tensor): Edge features, shape (n_edges, hidden_dim)
            edge_index (torch.Tensor): Edge connectivity, shape (2, n_edges)
        
        Returns:
            torch.Tensor: Updated node features, shape (n_nodes, hidden_dim)
        """
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # Gather features
        source_features = node_h[source_nodes]  # (n_edges, hidden_dim)
        target_features = node_h[target_nodes]  # (n_edges, hidden_dim)
        
        # Compute messages
        message_input = torch.cat([source_features, target_features, edge_h], dim=-1)
        messages = self.message_fn(message_input)  # (n_edges, hidden_dim)
        
        # Aggregate messages at target nodes
        aggregated = torch.zeros_like(node_h)
        aggregated.index_add_(0, target_nodes, messages)
        
        # Update node features
        update_input = torch.cat([node_h, aggregated], dim=-1)
        updated_h = self.update_fn(update_input)
        
        # Residual connection and layer normalization
        output = self.layer_norm(node_h + updated_h)
        
        return output
