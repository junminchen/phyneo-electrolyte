"""
Training script for Pairwise ML Potential Correction model.

This script trains the neural network to learn corrections to
physics-based pairwise potentials.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
from tqdm import tqdm

from phyneo.models import PairwiseMLCorrection


def load_data(data_path):
    """
    Load training data for pairwise correction.
    
    Args:
        data_path (str): Path to data file
    
    Returns:
        tuple: (pairwise_features, target_corrections)
    """
    # Example data loading - replace with actual data format
    data = np.load(data_path)
    features = torch.tensor(data["features"], dtype=torch.float32)
    corrections = torch.tensor(data["corrections"], dtype=torch.float32)
    return features, corrections


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Args:
        model: Pairwise correction model
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
    
    for features, target_corrections in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        target_corrections = target_corrections.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predicted_corrections = model(features)
        
        # Compute loss
        loss = criterion(predicted_corrections, target_corrections)
        
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
        model: Pairwise correction model
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
        for features, target_corrections in tqdm(dataloader, desc="Validation"):
            features = features.to(device)
            target_corrections = target_corrections.to(device)
            
            # Forward pass
            predicted_corrections = model(features)
            
            # Compute loss
            loss = criterion(predicted_corrections, target_corrections)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train Pairwise ML Correction")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--val_data_path", type=str, default=None,
                       help="Path to validation data")
    parser.add_argument("--input_dim", type=int, default=64,
                       help="Input feature dimension")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128, 64],
                       help="Hidden layer dimensions")
    parser.add_argument("--output_dim", type=int, default=1,
                       help="Output dimension")
    parser.add_argument("--activation", type=str, default="relu",
                       choices=["relu", "tanh", "elu"],
                       help="Activation function")
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
    train_features, train_corrections = load_data(args.data_path)
    train_dataset = TensorDataset(train_features, train_corrections)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.val_data_path:
        print("Loading validation data...")
        val_features, val_corrections = load_data(args.val_data_path)
        val_dataset = TensorDataset(val_features, val_corrections)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    else:
        val_loader = None
    
    # Initialize model
    print("Initializing model...")
    model = PairwiseMLCorrection(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
        activation=args.activation,
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
                checkpoint_path = os.path.join(args.output_dir, "best_pairwise_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"pairwise_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
