# API Reference

## Models

### SlaterTypeFunction

Slater-type orbital function for modeling short-range interactions.

```python
from phyneo.models import SlaterTypeFunction

model = SlaterTypeFunction(
    n_orbitals=32,        # Number of Slater orbitals
    principal_n=1,        # Principal quantum number
    learnable_zeta=True,  # Make orbital exponents learnable
    initial_zeta=1.0      # Initial value for zeta
)
```

**Methods:**

- `forward(distances)`: Compute Slater orbital values
  - Args: `distances` (torch.Tensor) - shape (batch, n_pairs)
  - Returns: torch.Tensor - shape (batch, n_pairs, n_orbitals)

- `get_cutoff_function(distances, cutoff_radius=5.0)`: Apply smooth cutoff
  - Args: `distances`, `cutoff_radius`
  - Returns: torch.Tensor - cutoff values between 0 and 1

### PairwiseMLCorrection

Neural network for learning pairwise potential corrections.

```python
from phyneo.models import PairwiseMLCorrection

model = PairwiseMLCorrection(
    input_dim=64,                      # Input feature dimension
    hidden_dims=[128, 128, 64],        # Hidden layer sizes
    output_dim=1,                      # Output dimension
    activation="relu",                 # Activation function
    use_batch_norm=True,               # Use batch normalization
    dropout_rate=0.1                   # Dropout rate
)
```

**Methods:**

- `forward(pairwise_features)`: Forward pass through network
  - Args: `pairwise_features` - shape (batch, n_pairs, input_dim)
  - Returns: torch.Tensor - shape (batch, n_pairs, output_dim)

- `get_energy_correction(pairwise_features, aggregate=True)`: Get total energy
  - Args: `pairwise_features`, `aggregate` (bool)
  - Returns: torch.Tensor - energy correction

### SubgraphNeuralNetwork

Graph neural network for molecular representations.

```python
from phyneo.models import SubgraphNeuralNetwork

model = SubgraphNeuralNetwork(
    node_input_dim=32,     # Node feature dimension
    edge_input_dim=16,     # Edge feature dimension
    hidden_dim=128,        # Hidden layer dimension
    output_dim=64,         # Output dimension
    n_layers=3,            # Number of message passing layers
    aggregation="sum",     # Graph aggregation method
    dropout_rate=0.1       # Dropout rate
)
```

**Methods:**

- `forward(node_features, edge_features, edge_index)`: Forward pass
  - Args: 
    - `node_features` - shape (n_nodes, node_input_dim)
    - `edge_features` - shape (n_edges, edge_input_dim)
    - `edge_index` - shape (2, n_edges)
  - Returns: torch.Tensor - shape (n_nodes, output_dim)

- `aggregate_graph(node_representations, batch_index=None)`: Aggregate to graph level
  - Args: `node_representations`, `batch_index`
  - Returns: torch.Tensor - graph-level representation

## Utilities

### Data Processing

```python
from phyneo.utils import (
    compute_pairwise_distances,
    build_neighbor_list,
    compute_coulomb_features,
    atoms_to_graph,
    normalize_features
)
```

**compute_pairwise_distances(positions, cell=None, pbc=None)**

Compute pairwise distances between atoms.

- Args:
  - `positions` (np.ndarray): Atomic positions, shape (n_atoms, 3)
  - `cell` (np.ndarray): Unit cell vectors (optional)
  - `pbc` (list): Periodic boundary conditions (optional)
- Returns: np.ndarray - distance matrix (n_atoms, n_atoms)

**build_neighbor_list(atoms, cutoff=5.0)**

Build neighbor list for atoms within cutoff distance.

- Args:
  - `atoms` (ase.Atoms): ASE Atoms object
  - `cutoff` (float): Cutoff distance
- Returns: dict - neighbor list {atom_idx: [neighbor_indices]}

**compute_coulomb_features(positions, charges, epsilon=1.0)**

Compute Coulomb interaction features.

- Args:
  - `positions` (np.ndarray): Atomic positions
  - `charges` (np.ndarray): Atomic charges
  - `epsilon` (float): Dielectric constant
- Returns: np.ndarray - Coulomb interaction matrix

**atoms_to_graph(atoms, cutoff=5.0)**

Convert ASE Atoms to graph representation.

- Args:
  - `atoms` (ase.Atoms): ASE Atoms object
  - `cutoff` (float): Cutoff for edges
- Returns: dict with 'node_features', 'edge_features', 'edge_index'

**normalize_features(features, mean=None, std=None)**

Normalize features using z-score normalization.

- Args:
  - `features` (np.ndarray): Features to normalize
  - `mean`, `std`: Normalization parameters (computed if None)
- Returns: tuple (normalized_features, mean, std)

### MD Simulation

```python
from phyneo.utils import MDSimulator, analyze_trajectory, save_snapshot
```

**MDSimulator**

Molecular dynamics simulator using PhyNEO force field.

```python
simulator = MDSimulator(
    atoms,              # ASE Atoms object
    calculator,         # Calculator for forces/energies
    temperature=300.0,  # Target temperature (K)
    timestep=1.0,       # MD timestep (fs)
    friction=0.01       # Friction coefficient
)

# Run simulation
results = simulator.run(
    nsteps=1000,
    trajectory_file='trajectory.traj',
    log_interval=10
)
```

**Methods:**

- `run(nsteps, trajectory_file=None, log_interval=10)`: Run MD
  - Returns: dict with 'energies', 'temperatures'

- `get_trajectory()`: Get current atomic configuration
- `set_temperature(temperature)`: Set new target temperature

**analyze_trajectory(trajectory_file)**

Analyze MD trajectory.

- Args: `trajectory_file` (str): Path to trajectory
- Returns: dict with analysis results

**save_snapshot(atoms, filename, format='xyz')**

Save atomic snapshot.

- Args: `atoms`, `filename`, `format`

## Training

All training scripts accept command-line arguments:

### train_slater.py

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

### train_pairwise.py

```bash
python phyneo/training/train_pairwise.py \
    --data_path data/pairwise_training.npz \
    --val_data_path data/pairwise_validation.npz \
    --input_dim 64 \
    --hidden_dims 128 128 64 \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.1 \
    --weight_decay 1e-5 \
    --output_dir checkpoints/pairwise \
    --device cuda
```

### train_subgraph.py

```bash
python phyneo/training/train_subgraph.py \
    --data_path data/graph_training.npz \
    --val_data_path data/graph_validation.npz \
    --node_input_dim 32 \
    --edge_input_dim 16 \
    --hidden_dim 128 \
    --n_layers 3 \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --dropout 0.1 \
    --output_dir checkpoints/subgraph \
    --device cuda
```

## Examples

### Loading a Trained Model

```python
import torch
from phyneo.models import SlaterTypeFunction

# Initialize model
model = SlaterTypeFunction(n_orbitals=32)

# Load checkpoint
checkpoint = torch.load('checkpoints/best_slater_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use model
with torch.no_grad():
    output = model(distances)
```

### Running MD Simulation

```python
from ase import Atoms
from phyneo.utils import MDSimulator

# Create system
atoms = Atoms(...)

# Create calculator (implement your own)
calculator = MyPhyNEOCalculator(models)

# Setup simulator
simulator = MDSimulator(
    atoms=atoms,
    calculator=calculator,
    temperature=300.0,
    timestep=1.0
)

# Run simulation
results = simulator.run(
    nsteps=10000,
    trajectory_file='md.traj',
    log_interval=100
)

print(f"Average temperature: {results['temperatures'].mean():.1f} K")
```

### Processing Molecular Graph

```python
from ase.io import read
from phyneo.utils import atoms_to_graph

# Load structure
atoms = read('structure.xyz')

# Convert to graph
graph = atoms_to_graph(atoms, cutoff=5.0)

print(f"Nodes: {graph['node_features'].shape}")
print(f"Edges: {graph['edge_features'].shape}")
```

## Data Formats

### Slater Training Data (.npz)

```python
{
    'distances': np.ndarray,  # Shape: (n_samples, n_pairs)
    'energies': np.ndarray    # Shape: (n_samples, n_pairs)
}
```

### Pairwise Training Data (.npz)

```python
{
    'features': np.ndarray,     # Shape: (n_samples, feature_dim)
    'corrections': np.ndarray   # Shape: (n_samples, output_dim)
}
```

### Graph Training Data (.npz)

```python
{
    'node_features': object array,  # List of (n_nodes, node_dim) arrays
    'edge_features': object array,  # List of (n_edges, edge_dim) arrays
    'edge_indices': object array,   # List of (2, n_edges) arrays
    'targets': np.ndarray          # Shape: (n_graphs, target_dim)
}
```
