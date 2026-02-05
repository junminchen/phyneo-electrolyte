"""
Training script for Slater-Type Function model.

This script trains the Slater-type orbital function to fit
short-range interactions in liquid electrolytes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
from tqdm import tqdm

from phyneo.models import SlaterTypeFunction


def load_data(data_path):
    """
    Load training data for Slater function.
    
    Args:
        data_path (str): Path to data file
    
    Returns:
        tuple: (distances, target_energies)
    """
    # Example data loading - replace with actual data format
    data = np.load(data_path)
    distances = torch.tensor(data["distances"], dtype=torch.float32)
    energies = torch.tensor(data["energies"], dtype=torch.float32)
    return distances, energies


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Args:
        model: Slater function model
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
    
    for distances, target_energies in tqdm(dataloader, desc="Training"):
        distances = distances.to(device)
        target_energies = target_energies.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        slater_values = model(distances)
        
        # Aggregate over orbitals (sum or weighted sum)
        predicted_energies = slater_values.sum(dim=-1)
        
        # Compute loss
        loss = criterion(predicted_energies, target_energies)
        
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
        model: Slater function model
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
        for distances, target_energies in tqdm(dataloader, desc="Validation"):
            distances = distances.to(device)
            target_energies = target_energies.to(device)
            
            # Forward pass
            slater_values = model(distances)
            predicted_energies = slater_values.sum(dim=-1)
            
            # Compute loss
            loss = criterion(predicted_energies, target_energies)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train Slater-Type Function")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--val_data_path", type=str, default=None,
                       help="Path to validation data")
    parser.add_argument("--n_orbitals", type=int, default=32,
                       help="Number of Slater orbitals")
    parser.add_argument("--principal_n", type=int, default=1,
                       help="Principal quantum number")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
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
    train_distances, train_energies = load_data(args.data_path)
    train_dataset = TensorDataset(train_distances, train_energies)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.val_data_path:
        print("Loading validation data...")
        val_distances, val_energies = load_data(args.val_data_path)
        val_dataset = TensorDataset(val_distances, val_energies)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    else:
        val_loader = None
    
    # Initialize model
    print("Initializing model...")
    model = SlaterTypeFunction(
        n_orbitals=args.n_orbitals,
        principal_n=args.principal_n,
        learnable_zeta=True
    )
    model = model.to(device)
    print(model)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
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
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.output_dir, "best_slater_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"slater_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
