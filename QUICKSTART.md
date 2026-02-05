# PhyNEO-Electrolyte Quick Reference

## Installation

```bash
git clone https://github.com/junminchen/phyneo-electrolyte.git
cd phyneo-electrolyte
pip install -r requirements.txt
pip install -e .
```

## Generate Example Data

```bash
python examples/generate_example_data.py
```

This creates:
- `data/slater_training.npz` & `data/slater_validation.npz`
- `data/pairwise_training.npz` & `data/pairwise_validation.npz`
- `data/graph_training.npz` & `data/graph_validation.npz`

## Training Models

### 1. Slater-Type Function
```bash
python phyneo/training/train_slater.py \
    --data_path data/slater_training.npz \
    --val_data_path data/slater_validation.npz \
    --n_orbitals 32 \
    --epochs 100 \
    --device cuda
```

### 2. Pairwise Correction
```bash
python phyneo/training/train_pairwise.py \
    --data_path data/pairwise_training.npz \
    --val_data_path data/pairwise_validation.npz \
    --input_dim 64 \
    --hidden_dims 128 128 64 \
    --epochs 100 \
    --device cuda
```

### 3. Sub-graph Network
```bash
python phyneo/training/train_subgraph.py \
    --data_path data/graph_training.npz \
    --val_data_path data/graph_validation.npz \
    --node_input_dim 32 \
    --edge_input_dim 16 \
    --hidden_dim 128 \
    --n_layers 3 \
    --epochs 100 \
    --device cuda
```

## Run MD Example

```bash
cd examples/md_simulation
python run_md_example.py
```

Outputs:
- `initial_structure.xyz` - Initial configuration
- `final_structure.xyz` - Final configuration
- `trajectory.traj` - Full MD trajectory

## Using Models in Code

### Load Slater Model
```python
import torch
from phyneo.models import SlaterTypeFunction

model = SlaterTypeFunction(n_orbitals=32)
checkpoint = torch.load('checkpoints/best_slater_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Load Pairwise Model
```python
from phyneo.models import PairwiseMLCorrection

model = PairwiseMLCorrection(input_dim=64, hidden_dims=[128, 128, 64])
checkpoint = torch.load('checkpoints/best_pairwise_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Load Graph Model
```python
from phyneo.models import SubgraphNeuralNetwork

model = SubgraphNeuralNetwork(
    node_input_dim=32,
    edge_input_dim=16,
    hidden_dim=128,
    n_layers=3
)
checkpoint = torch.load('checkpoints/best_subgraph_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Data Processing

### Convert Structure to Graph
```python
from ase.io import read
from phyneo.utils import atoms_to_graph

atoms = read('structure.xyz')
graph = atoms_to_graph(atoms, cutoff=5.0)
```

### Compute Pairwise Features
```python
from phyneo.utils import compute_pairwise_distances, compute_coulomb_features

distances = compute_pairwise_distances(positions)
coulomb = compute_coulomb_features(positions, charges)
```

## MD Simulation

```python
from phyneo.utils import MDSimulator
from ase import Atoms

# Create system
atoms = Atoms(...)

# Setup simulator
simulator = MDSimulator(
    atoms=atoms,
    calculator=your_calculator,
    temperature=300.0,
    timestep=1.0,
    friction=0.01
)

# Run
results = simulator.run(
    nsteps=10000,
    trajectory_file='trajectory.traj',
    log_interval=100
)
```

## Common Parameters

### Model Sizes
- **Small**: n_orbitals=16, hidden_dim=64
- **Medium**: n_orbitals=32, hidden_dim=128
- **Large**: n_orbitals=64, hidden_dim=256

### Training
- **Learning Rate**: 0.001 (typical)
- **Batch Size**: 32 (adjust for GPU memory)
- **Epochs**: 100-200 (depends on convergence)
- **Dropout**: 0.1-0.2
- **Weight Decay**: 1e-5

### MD Simulation
- **Temperature**: 300 K (room temp)
- **Timestep**: 0.5-2.0 fs
- **Friction**: 0.001-0.01 (for Langevin)

## File Locations

```
phyneo-electrolyte/
├── phyneo/                  # Main package
│   ├── models/             # Neural network models
│   ├── training/           # Training scripts
│   └── utils/              # Utilities
├── examples/               # Examples
│   ├── md_simulation/      # MD example
│   └── generate_example_data.py
├── docs/                   # Documentation
├── data/                   # Training data (generated)
└── checkpoints/           # Model checkpoints (training output)
```

## Help & Documentation

- **README**: Overview and quick start
- **docs/INSTALLATION.md**: Installation guide
- **docs/TRAINING.md**: Training guide
- **docs/API.md**: Complete API reference
- **CONTRIBUTING.md**: Contribution guidelines
- **examples/**: Working examples

## Troubleshooting

### Import Error
```bash
pip install -e .
```

### CUDA Out of Memory
- Reduce batch size: `--batch_size 16`
- Use smaller model: `--hidden_dim 64`
- Use CPU: `--device cpu`

### Training Not Converging
- Reduce learning rate: `--lr 0.0001`
- Add more data
- Normalize features
- Check data quality

## Support

- GitHub Issues: https://github.com/junminchen/phyneo-electrolyte/issues
- Documentation: See `docs/` folder
- Examples: See `examples/` folder
