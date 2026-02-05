"""
Slater-Type Function for short-range interactions.

This module implements Slater-type orbitals for modeling short-range
electron-electron interactions in liquid electrolytes.
"""

import torch
import torch.nn as nn
import numpy as np


class SlaterTypeFunction(nn.Module):
    """
    Slater-Type Orbital (STO) function for short-range interactions.
    
    The Slater function has the form:
    φ(r) = N * r^(n-1) * exp(-ζ*r)
    
    where:
    - N is the normalization constant
    - n is the principal quantum number
    - ζ (zeta) is the orbital exponent
    - r is the distance
    
    Args:
        n_orbitals (int): Number of Slater orbitals
        principal_n (int): Principal quantum number (default: 1)
        learnable_zeta (bool): Whether to make zeta learnable (default: True)
        initial_zeta (float): Initial value for zeta (default: 1.0)
    """
    
    def __init__(
        self,
        n_orbitals=32,
        principal_n=1,
        learnable_zeta=True,
        initial_zeta=1.0
    ):
        super(SlaterTypeFunction, self).__init__()
        
        self.n_orbitals = n_orbitals
        self.principal_n = principal_n
        
        # Initialize orbital exponents
        if learnable_zeta:
            self.zeta = nn.Parameter(
                torch.ones(n_orbitals) * initial_zeta
            )
        else:
            self.register_buffer(
                "zeta",
                torch.ones(n_orbitals) * initial_zeta
            )
    
    def _compute_normalization(self, zeta):
        """Compute normalization constants for Slater orbitals."""
        n = self.principal_n
        # Normalization: N = (2*ζ)^(n+0.5) / sqrt((2n)!)
        # Simplified for common cases
        norm = torch.pow(2.0 * zeta, n + 0.5)
        if n == 1:
            norm = norm / np.sqrt(2.0)
        elif n == 2:
            norm = norm / np.sqrt(24.0)
        else:
            norm = norm / np.sqrt(np.math.factorial(2 * n))
        
        return norm
    
    def forward(self, distances):
        """
        Compute Slater-type orbital values for given distances.
        
        Args:
            distances (torch.Tensor): Pairwise distances, shape (batch, n_pairs)
        
        Returns:
            torch.Tensor: Slater orbital values, shape (batch, n_pairs, n_orbitals)
        """
        # Expand dimensions for broadcasting
        r = distances.unsqueeze(-1)  # (batch, n_pairs, 1)
        zeta = self.zeta.unsqueeze(0).unsqueeze(0)  # (1, 1, n_orbitals)
        
        # Compute normalization dynamically
        norm = self._compute_normalization(zeta)
        
        # Compute Slater function: φ(r) = N * r^(n-1) * exp(-ζ*r)
        if self.principal_n > 1:
            radial_part = torch.pow(r, self.principal_n - 1)
        else:
            radial_part = 1.0
        
        exponential_part = torch.exp(-zeta * r)
        
        slater_values = norm * radial_part * exponential_part
        
        return slater_values
    
    def get_cutoff_function(self, distances, cutoff_radius=5.0):
        """
        Apply a smooth cutoff function to the Slater orbitals.
        
        Args:
            distances (torch.Tensor): Pairwise distances
            cutoff_radius (float): Cutoff radius for interactions
        
        Returns:
            torch.Tensor: Cutoff values between 0 and 1
        """
        # Cosine cutoff function
        cutoff = 0.5 * (torch.cos(np.pi * distances / cutoff_radius) + 1.0)
        cutoff = torch.where(
            distances < cutoff_radius,
            cutoff,
            torch.zeros_like(cutoff)
        )
        return cutoff
    
    def extra_repr(self):
        """String representation of the module."""
        return (
            f"n_orbitals={self.n_orbitals}, "
            f"principal_n={self.principal_n}"
        )
