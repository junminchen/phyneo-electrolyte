# PhyNEO-Electrolyte

A Hybrid Physics-Driven Neural Network Force Field for Liquid Electrolytes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

PhyNEO-Electrolyte is a cutting-edge machine learning framework that combines physics-based models with neural networks to accurately simulate liquid electrolytes. The framework integrates:

- **Slater-Type Orbital Functions** for modeling short-range electron-electron interactions
- **Pairwise Machine Learning Corrections** to improve physics-based potentials
- **Sub-graph Neural Networks** for learning molecular representations

## Features

- 🔬 **Physics-Driven Architecture**: Combines traditional physics models with modern ML
- ⚡ **Efficient Training**: Modular design allows training each component independently
- 🧪 **MD Integration**: Seamless integration with ASE for molecular dynamics simulations
- 📊 **Flexible Models**: Configurable architectures for different electrolyte systems
- 🎯 **Accurate Predictions**: State-of-the-art accuracy for electrolyte properties

## Installation

### Basic Installation

```bash
git clone https://github.com/junminchen/phyneo-electrolyte.git
cd phyneo-electrolyte
pip install -r requirements.txt
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Train Slater-Type Function

```bash
python phyneo/training/train_slater.py \
    --data_path data/slater_training.npz \
    --n_orbitals 32 \
    --epochs 100 \
    --output_dir checkpoints/slater
```

### 2. Train Pairwise Correction

```bash
python phyneo/training/train_pairwise.py \
    --data_path data/pairwise_training.npz \
    --input_dim 64 \
    --hidden_dims 128 128 64 \
    --epochs 100 \
    --output_dir checkpoints/pairwise
```

### 3. Train Sub-graph Neural Network

```bash
python phyneo/training/train_subgraph.py \
    --data_path data/graph_training.npz \
    --node_input_dim 32 \
    --edge_input_dim 16 \
    --hidden_dim 128 \
    --n_layers 3 \
    --epochs 100 \
    --output_dir checkpoints/subgraph
```

### 4. Run MD Simulation

```bash
cd examples/md_simulation
python run_md_example.py
```

## Project Structure

```
phyneo-electrolyte/
├── phyneo/                      # Main package
│   ├── models/                  # Neural network models
│   │   ├── slater_function.py   # Slater-type orbital functions
│   │   ├── pairwise_correction.py  # Pairwise ML corrections
│   │   └── subgraph_network.py  # Graph neural networks
│   ├── training/                # Training scripts
│   │   ├── train_slater.py      # Train Slater functions
│   │   ├── train_pairwise.py    # Train pairwise corrections
│   │   └── train_subgraph.py    # Train graph networks
│   └── utils/                   # Utility functions
│       ├── data_utils.py        # Data processing
│       └── md_utils.py          # MD simulation utilities
├── examples/                    # Example scripts
│   └── md_simulation/           # MD simulation example
│       ├── run_md_example.py
│       └── README.md
├── data/                        # Data directory (user-provided)
├── docs/                        # Documentation
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Models

### Slater-Type Function

Models short-range interactions using Slater-type orbitals:

```python
from phyneo.models import SlaterTypeFunction

model = SlaterTypeFunction(
    n_orbitals=32,
    principal_n=1,
    learnable_zeta=True
)
```

### Pairwise ML Correction

Learns corrections to physics-based potentials:

```python
from phyneo.models import PairwiseMLCorrection

model = PairwiseMLCorrection(
    input_dim=64,
    hidden_dims=[128, 128, 64],
    output_dim=1
)
```

### Sub-graph Neural Network

Processes molecular graphs using message passing:

```python
from phyneo.models import SubgraphNeuralNetwork

model = SubgraphNeuralNetwork(
    node_input_dim=32,
    edge_input_dim=16,
    hidden_dim=128,
    n_layers=3
)
```

## Training

Each component can be trained independently:

1. **Slater Function Training**: Fit short-range orbital functions to quantum mechanical data
2. **Pairwise Correction Training**: Learn corrections to pair potentials
3. **Sub-graph Network Training**: Train graph neural networks on molecular structures

See `phyneo/training/` for detailed training scripts and options.

## Molecular Dynamics

Run MD simulations using the trained models:

```python
from phyneo.utils.md_utils import MDSimulator
from ase import Atoms

# Create system
atoms = Atoms(...)

# Setup and run simulation
simulator = MDSimulator(
    atoms=atoms,
    calculator=phyneo_calculator,
    temperature=300.0,
    timestep=1.0
)

results = simulator.run(nsteps=1000)
```

See `examples/md_simulation/` for complete examples.

## Data Format

### Slater Training Data
```python
# Save as .npz file
np.savez('slater_training.npz',
    distances=distances,  # Shape: (n_samples, n_pairs)
    energies=energies     # Shape: (n_samples, n_pairs)
)
```

### Pairwise Training Data
```python
np.savez('pairwise_training.npz',
    features=features,      # Shape: (n_samples, feature_dim)
    corrections=corrections # Shape: (n_samples, output_dim)
)
```

### Graph Training Data
```python
np.savez('graph_training.npz',
    node_features=node_features,  # List of arrays
    edge_features=edge_features,  # List of arrays
    edge_indices=edge_indices,    # List of arrays
    targets=targets               # Array of targets
)
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- ASE >= 3.22.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- matplotlib >= 3.3.0
- pandas >= 1.3.0
- tqdm >= 4.60.0

## Citation

If you use PhyNEO-Electrolyte in your research, please cite:

```bibtex
@software{phyneo_electrolyte,
  title={PhyNEO-Electrolyte: A Hybrid Physics-Driven Neural Network Force Field for Liquid Electrolytes},
  author={dreamchen},
  year={2026},
  url={https://github.com/junminchen/phyneo-electrolyte}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.

## Acknowledgments

This project builds upon:
- ASE (Atomic Simulation Environment)
- PyTorch deep learning framework
- Various quantum chemistry and ML research
