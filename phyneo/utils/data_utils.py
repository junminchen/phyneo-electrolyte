"""
Data processing utilities for PhyNEO-Electrolyte.
"""

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import NeighborList


def compute_pairwise_distances(positions, cell=None, pbc=None):
    """
    Compute pairwise distances between atoms.
    
    Args:
        positions (np.ndarray): Atomic positions, shape (n_atoms, 3)
        cell (np.ndarray): Unit cell vectors (optional)
        pbc (list): Periodic boundary conditions (optional)
    
    Returns:
        np.ndarray: Pairwise distance matrix, shape (n_atoms, n_atoms)
    """
    n_atoms = len(positions)
    distances = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            vec = positions[j] - positions[i]
            
            # Apply periodic boundary conditions if specified
            if cell is not None and pbc is not None:
                for k, periodic in enumerate(pbc):
                    if periodic:
                        vec[k] -= cell[k] * np.round(vec[k] / cell[k])
            
            dist = np.linalg.norm(vec)
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def build_neighbor_list(atoms, cutoff=5.0):
    """
    Build neighbor list for atoms within cutoff distance.
    
    Args:
        atoms (ase.Atoms): ASE Atoms object
        cutoff (float): Cutoff distance for neighbors
    
    Returns:
        dict: Neighbor list as {atom_idx: [neighbor_indices]}
    """
    cutoffs = [cutoff / 2.0] * len(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    
    neighbors = {}
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        neighbors[i] = indices.tolist()
    
    return neighbors


def compute_coulomb_features(positions, charges, epsilon=1.0):
    """
    Compute Coulomb interaction features.
    
    Args:
        positions (np.ndarray): Atomic positions
        charges (np.ndarray): Atomic charges
        epsilon (float): Dielectric constant
    
    Returns:
        np.ndarray: Coulomb interaction matrix
    """
    n_atoms = len(positions)
    coulomb = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[j] - positions[i])
            if dist > 1e-6:  # Avoid division by zero
                interaction = charges[i] * charges[j] / (dist * epsilon)
                coulomb[i, j] = interaction
                coulomb[j, i] = interaction
    
    return coulomb


def atoms_to_graph(atoms, cutoff=5.0):
    """
    Convert ASE Atoms object to graph representation.
    
    Args:
        atoms (ase.Atoms): ASE Atoms object
        cutoff (float): Cutoff distance for edges
    
    Returns:
        dict: Graph with node features, edge features, and edge indices
    """
    n_atoms = len(atoms)
    positions = atoms.get_positions()
    
    # Node features: atomic numbers (can be extended)
    node_features = atoms.get_atomic_numbers().reshape(-1, 1)
    
    # Build neighbor list
    neighbors = build_neighbor_list(atoms, cutoff)
    
    # Build edges
    edge_index = []
    edge_features = []
    
    for i in range(n_atoms):
        for j in neighbors[i]:
            if j > i:  # Avoid duplicates
                edge_index.append([i, j])
                edge_index.append([j, i])  # Add reverse edge
                
                # Edge features: distance
                dist = np.linalg.norm(positions[j] - positions[i])
                edge_features.append([dist])
                edge_features.append([dist])
    
    edge_index = np.array(edge_index).T if edge_index else np.zeros((2, 0))
    edge_features = np.array(edge_features) if edge_features else np.zeros((0, 1))
    
    return {
        'node_features': node_features,
        'edge_features': edge_features,
        'edge_index': edge_index
    }


def normalize_features(features, mean=None, std=None):
    """
    Normalize features using z-score normalization.
    
    Args:
        features (np.ndarray): Features to normalize
        mean (np.ndarray): Mean values (computed if None)
        std (np.ndarray): Standard deviation values (computed if None)
    
    Returns:
        tuple: (normalized_features, mean, std)
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
        std[std < 1e-8] = 1.0  # Avoid division by zero
    
    normalized = (features - mean) / std
    return normalized, mean, std
