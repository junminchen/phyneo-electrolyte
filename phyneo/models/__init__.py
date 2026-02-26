"""
Neural network models for PhyNEO-Electrolyte.

This module contains implementations of:
- Slater-type orbital functions for short-range interactions
- Pairwise ML potential corrections
- Sub-graph neural networks for molecular representations
"""

from .eapnn import EAPNNForce

__all__ = [
    "EAPNNForce",
]
