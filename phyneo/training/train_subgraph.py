"""
Training script for Sub-graph Neural Network model.

This script trains the graph neural network to learn
molecular representations from sub-graphs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import os
from tqdm import tqdm

from phyneo.models import SubgraphNeuralNetwork


class GraphDataset(Dataset):
    """
    Dataset for graph data.
    
    Args:
        data_path (str): Path to data file
    """
    
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        self.node_features = data["node_features"]
        self.edge_features = data["edge_features"]
        self.edge_indices = data["edge_indices"]
        self.targets = data["targets"]
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'node_features': torch.tensor(self.node_features[idx], dtype=torch.float32),
            'edge_features': torch.tensor(self.edge_features[idx], dtype=torch.float32),
            'edge_index': torch.tensor(self.edge_indices[idx], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Args:
        model: Sub-graph neural network model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
    
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        node_features = batch['node_features'].to(device)
        edge_features = batch['edge_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        node_representations = model(node_features, edge_features, edge_index)
        
        # Aggregate to graph-level representation
        graph_representation = model.aggregate_graph(node_representations)
        
        # Compute loss
        loss = criterion(graph_representation, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: Sub-graph neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            node_features = batch['node_features'].to(device)
            edge_features = batch['edge_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            node_representations = model(node_features, edge_features, edge_index)
            graph_representation = model.aggregate_graph(node_representations)
            
            # Compute loss
            loss = criterion(graph_representation, targets)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train Sub-graph Neural Network")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--val_data_path", type=str, default=None,
                       help="Path to validation data")
    parser.add_argument("--node_input_dim", type=int, default=32,
                       help="Node input feature dimension")
    parser.add_argument("--edge_input_dim", type=int, default=16,
                       help="Edge input feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden layer dimension")
    parser.add_argument("--output_dim", type=int, default=64,
                       help="Output dimension")
    parser.add_argument("--n_layers", type=int, default=3,
                       help="Number of message passing layers")
    parser.add_argument("--aggregation", type=str, default="sum",
                       choices=["sum", "mean", "max"],
                       help="Graph aggregation method")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to train on (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading training data...")
    train_dataset = GraphDataset(args.data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.val_data_path:
        print("Loading validation data...")
        val_dataset = GraphDataset(args.val_data_path)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    else:
        val_loader = None
    
    # Initialize model
    print("Initializing model...")
    model = SubgraphNeuralNetwork(
        node_input_dim=args.node_input_dim,
        edge_input_dim=args.edge_input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        n_layers=args.n_layers,
        aggregation=args.aggregation,
        dropout_rate=args.dropout
    )
    model = model.to(device)
    print(model)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss:.6f}")
        
        # Validate
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.output_dir, "best_subgraph_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"subgraph_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
