"""
Generate example training data for PhyNEO-Electrolyte models.

This script creates synthetic training data for demonstration purposes.
In practice, you would use real quantum mechanical calculations.
"""

import numpy as np
import os


def generate_slater_data(n_samples=1000, n_pairs=50, output_dir='data'):
    """
    Generate synthetic training data for Slater-Type Function.
    
    Args:
        n_samples (int): Number of training samples
        n_pairs (int): Number of atomic pairs per sample
        output_dir (str): Output directory
    """
    print("Generating Slater function training data...")
    
    # Generate distances (typically 1-5 Angstroms)
    distances = np.random.uniform(1.0, 5.0, size=(n_samples, n_pairs))
    
    # Generate synthetic energies using a simple exponential decay
    # In practice, these would come from quantum mechanical calculations
    energies = np.zeros_like(distances)
    for i in range(n_samples):
        for j in range(n_pairs):
            r = distances[i, j]
            # Simple Slater-like function for demo
            energies[i, j] = np.exp(-2.0 * r) + np.random.normal(0, 0.01)
    
    # Save training data
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, 'slater_training.npz')
    np.savez(train_file, distances=distances[:800], energies=energies[:800])
    print(f"Saved training data: {train_file}")
    
    # Save validation data
    val_file = os.path.join(output_dir, 'slater_validation.npz')
    np.savez(val_file, distances=distances[800:], energies=energies[800:])
    print(f"Saved validation data: {val_file}")
    
    print(f"Generated {n_samples} samples with {n_pairs} pairs each")


def generate_pairwise_data(n_samples=1000, feature_dim=64, output_dir='data'):
    """
    Generate synthetic training data for Pairwise ML Correction.
    
    Args:
        n_samples (int): Number of training samples
        feature_dim (int): Dimension of input features
        output_dir (str): Output directory
    """
    print("\nGenerating pairwise correction training data...")
    
    # Generate random features (could be distances, charges, etc.)
    features = np.random.randn(n_samples, feature_dim)
    
    # Generate synthetic corrections
    # In practice, these would be residuals from physics-based models
    corrections = np.zeros((n_samples, 1))
    for i in range(n_samples):
        # Simple linear model with noise for demo
        corrections[i] = (
            np.sum(features[i, :10] * 0.1) + 
            np.random.normal(0, 0.05)
        )
    
    # Save training data
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, 'pairwise_training.npz')
    np.savez(train_file, 
             features=features[:800], 
             corrections=corrections[:800])
    print(f"Saved training data: {train_file}")
    
    # Save validation data
    val_file = os.path.join(output_dir, 'pairwise_validation.npz')
    np.savez(val_file, 
             features=features[800:], 
             corrections=corrections[800:])
    print(f"Saved validation data: {val_file}")
    
    print(f"Generated {n_samples} samples with {feature_dim} features each")


def generate_graph_data(n_graphs=200, output_dir='data'):
    """
    Generate synthetic training data for Sub-graph Neural Network.
    
    Args:
        n_graphs (int): Number of molecular graphs
        output_dir (str): Output directory
    """
    print("\nGenerating graph neural network training data...")
    
    node_features_list = []
    edge_features_list = []
    edge_indices_list = []
    targets = np.zeros(n_graphs)
    
    for i in range(n_graphs):
        # Random graph size (5-15 atoms)
        n_atoms = np.random.randint(5, 16)
        
        # Node features (e.g., atomic numbers, charges)
        # Using 32-dimensional features
        node_features = np.random.randn(n_atoms, 32).astype(np.float32)
        
        # Generate random edges (molecular bonds)
        n_edges = np.random.randint(n_atoms, n_atoms * 2)
        edge_index = np.zeros((2, n_edges), dtype=np.int64)
        edge_features = np.zeros((n_edges, 16), dtype=np.float32)
        
        for j in range(n_edges):
            # Random edge between two atoms
            src = np.random.randint(0, n_atoms)
            dst = np.random.randint(0, n_atoms)
            if src != dst:
                edge_index[0, j] = src
                edge_index[1, j] = dst
                # Edge features (e.g., bond distances)
                edge_features[j] = np.random.randn(16).astype(np.float32)
        
        node_features_list.append(node_features)
        edge_features_list.append(edge_features)
        edge_indices_list.append(edge_index)
        
        # Target (e.g., total energy)
        targets[i] = np.sum(node_features[:, 0]) + np.random.normal(0, 0.1)
    
    # Save training data
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, 'graph_training.npz')
    
    # Convert lists to object arrays properly
    train_node_features = np.empty(160, dtype=object)
    train_edge_features = np.empty(160, dtype=object)
    train_edge_indices = np.empty(160, dtype=object)
    for i in range(160):
        train_node_features[i] = node_features_list[i]
        train_edge_features[i] = edge_features_list[i]
        train_edge_indices[i] = edge_indices_list[i]
    
    np.savez(train_file,
             node_features=train_node_features,
             edge_features=train_edge_features,
             edge_indices=train_edge_indices,
             targets=targets[:160])
    print(f"Saved training data: {train_file}")
    
    # Save validation data
    val_file = os.path.join(output_dir, 'graph_validation.npz')
    
    val_node_features = np.empty(40, dtype=object)
    val_edge_features = np.empty(40, dtype=object)
    val_edge_indices = np.empty(40, dtype=object)
    for i in range(40):
        val_node_features[i] = node_features_list[160 + i]
        val_edge_features[i] = edge_features_list[160 + i]
        val_edge_indices[i] = edge_indices_list[160 + i]
    
    np.savez(val_file,
             node_features=val_node_features,
             edge_features=val_edge_features,
             edge_indices=val_edge_indices,
             targets=targets[160:])
    print(f"Saved validation data: {val_file}")
    
    print(f"Generated {n_graphs} molecular graphs")


def main():
    """Generate all example training data."""
    print("=" * 70)
    print("PhyNEO-Electrolyte: Generate Example Training Data")
    print("=" * 70)
    print("\nNOTE: This generates SYNTHETIC data for demonstration only.")
    print("For real applications, use quantum mechanical calculations.\n")
    
    # Create data directory
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate data for each model
    generate_slater_data(n_samples=1000, n_pairs=50, output_dir=data_dir)
    generate_pairwise_data(n_samples=1000, feature_dim=64, output_dir=data_dir)
    generate_graph_data(n_graphs=200, output_dir=data_dir)
    
    print("\n" + "=" * 70)
    print("Data generation completed!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  {data_dir}/slater_training.npz")
    print(f"  {data_dir}/slater_validation.npz")
    print(f"  {data_dir}/pairwise_training.npz")
    print(f"  {data_dir}/pairwise_validation.npz")
    print(f"  {data_dir}/graph_training.npz")
    print(f"  {data_dir}/graph_validation.npz")
    print("\nYou can now train models using these data files:")
    print("  python phyneo/training/train_slater.py --data_path data/slater_training.npz")
    print("  python phyneo/training/train_pairwise.py --data_path data/pairwise_training.npz")
    print("  python phyneo/training/train_subgraph.py --data_path data/graph_training.npz")


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    main()
