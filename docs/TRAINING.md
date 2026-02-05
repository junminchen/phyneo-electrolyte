# Training Guide

This guide provides detailed instructions for training PhyNEO-Electrolyte models.

## Overview

PhyNEO-Electrolyte consists of three main components that can be trained independently:

1. **Slater-Type Function**: Models short-range interactions
2. **Pairwise ML Correction**: Learns corrections to physics-based potentials
3. **Sub-graph Neural Network**: Learns molecular representations

## Data Preparation

### 1. Slater Function Data

Format: NumPy `.npz` file with:
- `distances`: Pairwise distances, shape `(n_samples, n_pairs)`
- `energies`: Target energies, shape `(n_samples, n_pairs)`

Example:
```python
import numpy as np

# Generate or load your data
distances = np.random.rand(1000, 100) * 5.0  # Random distances
energies = np.random.rand(1000, 100)  # Target energies

# Save
np.savez('data/slater_training.npz', 
         distances=distances, 
         energies=energies)
```

### 2. Pairwise Correction Data

Format: NumPy `.npz` file with:
- `features`: Input features, shape `(n_samples, feature_dim)`
- `corrections`: Target corrections, shape `(n_samples, output_dim)`

Example:
```python
features = np.random.rand(1000, 64)  # Pairwise features
corrections = np.random.rand(1000, 1)  # Energy corrections

np.savez('data/pairwise_training.npz',
         features=features,
         corrections=corrections)
```

### 3. Graph Network Data

Format: NumPy `.npz` file with:
- `node_features`: List of node feature arrays
- `edge_features`: List of edge feature arrays
- `edge_indices`: List of edge index arrays
- `targets`: Target values

Example:
```python
# Example for 100 graphs
node_features = [np.random.rand(10, 32) for _ in range(100)]
edge_features = [np.random.rand(20, 16) for _ in range(100)]
edge_indices = [np.random.randint(0, 10, (2, 20)) for _ in range(100)]
targets = np.random.rand(100, 64)

np.savez('data/graph_training.npz',
         node_features=node_features,
         edge_features=edge_features,
         edge_indices=edge_indices,
         targets=targets)
```

## Training Slater-Type Function

### Basic Training

```bash
python phyneo/training/train_slater.py \
    --data_path data/slater_training.npz \
    --val_data_path data/slater_validation.npz \
    --n_orbitals 32 \
    --principal_n 1 \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --output_dir checkpoints/slater \
    --device cuda
```

### Key Parameters

- `--n_orbitals`: Number of Slater orbitals (default: 32)
- `--principal_n`: Principal quantum number (default: 1)
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)

### Tips

- Start with fewer orbitals (16-32) for faster training
- Use validation data to monitor overfitting
- Adjust learning rate if training is unstable

## Training Pairwise Correction

### Basic Training

```bash
python phyneo/training/train_pairwise.py \
    --data_path data/pairwise_training.npz \
    --val_data_path data/pairwise_validation.npz \
    --input_dim 64 \
    --hidden_dims 128 128 64 \
    --output_dim 1 \
    --activation relu \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.1 \
    --weight_decay 1e-5 \
    --output_dir checkpoints/pairwise \
    --device cuda
```

### Key Parameters

- `--input_dim`: Input feature dimension (default: 64)
- `--hidden_dims`: Hidden layer sizes (default: [128, 128, 64])
- `--activation`: Activation function (relu/tanh/elu)
- `--dropout`: Dropout rate (default: 0.1)
- `--weight_decay`: L2 regularization (default: 1e-5)

### Tips

- Increase hidden dimensions for complex systems
- Use batch normalization for stable training
- Monitor validation loss for early stopping

## Training Sub-graph Neural Network

### Basic Training

```bash
python phyneo/training/train_subgraph.py \
    --data_path data/graph_training.npz \
    --val_data_path data/graph_validation.npz \
    --node_input_dim 32 \
    --edge_input_dim 16 \
    --hidden_dim 128 \
    --output_dim 64 \
    --n_layers 3 \
    --aggregation sum \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.1 \
    --weight_decay 1e-5 \
    --output_dir checkpoints/subgraph \
    --device cuda
```

### Key Parameters

- `--node_input_dim`: Node feature dimension (default: 32)
- `--edge_input_dim`: Edge feature dimension (default: 16)
- `--hidden_dim`: Hidden layer dimension (default: 128)
- `--n_layers`: Number of message passing layers (default: 3)
- `--aggregation`: Graph pooling method (sum/mean/max)

### Tips

- More layers capture longer-range interactions
- Use edge features for distance and bond information
- Experiment with different aggregation methods

## Monitoring Training

All training scripts print progress:

```
Epoch 1/100
Training: 100%|████████| 32/32 [00:10<00:00]
Training Loss: 0.123456
Validation Loss: 0.234567
Saved best model to checkpoints/best_model.pth
```

### Checkpoints

Models are automatically saved:
- `best_<model>_model.pth`: Best model based on validation loss
- `<model>_model_epoch_N.pth`: Periodic checkpoints every 10 epochs

### Loading Checkpoints

```python
import torch
from phyneo.models import SlaterTypeFunction

# Load model
model = SlaterTypeFunction(n_orbitals=32)
checkpoint = torch.load('checkpoints/best_slater_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Advanced Training

### Using Custom Data Loaders

```python
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path):
        # Custom data loading
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Use in training
train_loader = DataLoader(CustomDataset('data.npz'), batch_size=32)
```

### Transfer Learning

Start from pretrained model:

```python
# Load pretrained model
pretrained = torch.load('pretrained_model.pth')
model.load_state_dict(pretrained['model_state_dict'])

# Fine-tune on new data
# ... training code ...
```

### Distributed Training

For multi-GPU training, use PyTorch distributed:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DistributedDataParallel(model)
```

## Best Practices

1. **Data Quality**: Ensure high-quality reference data (QM calculations)
2. **Validation Split**: Use 10-20% of data for validation
3. **Hyperparameter Tuning**: Use grid search or Bayesian optimization
4. **Early Stopping**: Stop training when validation loss plateaus
5. **Learning Rate Schedule**: Reduce LR when loss plateaus
6. **Regularization**: Use dropout and weight decay to prevent overfitting

## Troubleshooting

### Training Loss Not Decreasing

- Reduce learning rate
- Check data normalization
- Increase model capacity
- Verify data quality

### Validation Loss Increasing

- Reduce model complexity
- Increase dropout/weight decay
- Get more training data
- Check for data leakage

### Out of Memory Errors

- Reduce batch size
- Use gradient accumulation
- Use mixed precision training
- Simplify model architecture

## Next Steps

After training:
1. Evaluate models on test set
2. Integrate models into MD calculator
3. Run production MD simulations
4. Analyze results

See the [MD Example](../examples/md_simulation/README.md) for integration examples.
