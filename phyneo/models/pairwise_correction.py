"""
Pairwise Machine Learning Potential Correction.

This module implements a neural network to learn corrections to
physics-based pairwise potentials.
"""

import torch
import torch.nn as nn


class PairwiseMLCorrection(nn.Module):
    """
    Neural network for learning pairwise potential corrections.
    
    This module takes pairwise features (distances, Coulomb interactions, etc.)
    and learns corrections to the physics-based potential energy.
    
    Args:
        input_dim (int): Dimension of input features (default: 64)
        hidden_dims (list): List of hidden layer dimensions (default: [128, 128, 64])
        output_dim (int): Output dimension (default: 1, for energy correction)
        activation (str): Activation function ('relu', 'tanh', 'elu') (default: 'relu')
        use_batch_norm (bool): Whether to use batch normalization (default: True)
        dropout_rate (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim=64,
        hidden_dims=None,
        output_dim=1,
        activation="relu",
        use_batch_norm=True,
        dropout_rate=0.1
    ):
        super(PairwiseMLCorrection, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Select activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.activation)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, pairwise_features):
        """
        Forward pass through the correction network.
        
        Args:
            pairwise_features (torch.Tensor): Input features, shape (batch, n_pairs, input_dim)
                                             or (batch, input_dim)
        
        Returns:
            torch.Tensor: Energy corrections, shape (batch, n_pairs, output_dim) or (batch, output_dim)
        """
        original_shape = pairwise_features.shape
        
        # Flatten if 3D input
        if len(original_shape) == 3:
            batch_size, n_pairs, feat_dim = original_shape
            pairwise_features = pairwise_features.view(-1, feat_dim)
            
            corrections = self.network(pairwise_features)
            corrections = corrections.view(batch_size, n_pairs, -1)
        else:
            corrections = self.network(pairwise_features)
        
        return corrections
    
    def get_energy_correction(self, pairwise_features, aggregate=True):
        """
        Compute total energy correction for a system.
        
        Args:
            pairwise_features (torch.Tensor): Pairwise features
            aggregate (bool): Whether to sum over all pairs (default: True)
        
        Returns:
            torch.Tensor: Total energy correction
        """
        corrections = self.forward(pairwise_features)
        
        if aggregate:
            # Sum over all pairs to get total correction
            if len(corrections.shape) == 3:
                corrections = corrections.sum(dim=1)  # Sum over pairs
        
        return corrections
    
    def extra_repr(self):
        """String representation of the module."""
        return (
            f"input_dim={self.input_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"output_dim={self.output_dim}"
        )
