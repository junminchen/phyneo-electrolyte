# Changelog

All notable changes to PhyNEO-Electrolyte will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-05

### Added

#### Core Models
- Slater-Type Function model for short-range interactions
  - Configurable number of orbitals and principal quantum numbers
  - Learnable orbital exponents
  - Smooth cutoff functions
- Pairwise ML Correction model
  - Flexible neural network architecture
  - Batch normalization and dropout
  - Energy correction aggregation
- Sub-graph Neural Network model
  - Message passing architecture
  - Node and edge feature processing
  - Graph-level aggregation (sum/mean/max)

#### Training Scripts
- `train_slater.py` - Training script for Slater-type functions
- `train_pairwise.py` - Training script for pairwise corrections
- `train_subgraph.py` - Training script for graph neural networks
- Support for:
  - CPU and CUDA training
  - Validation monitoring
  - Checkpoint saving
  - Learning rate scheduling
  - Progress tracking with tqdm

#### Utilities
- Data processing utilities:
  - Pairwise distance computation
  - Neighbor list building
  - Coulomb feature calculation
  - Graph conversion from ASE atoms
  - Feature normalization
- MD simulation utilities:
  - MDSimulator class for running simulations
  - Langevin and Verlet integrators
  - Trajectory analysis
  - Snapshot saving

#### Examples
- MD simulation example with liquid electrolyte
- Example data generation script
- Complete workflow demonstration

#### Documentation
- Comprehensive README with quick start guide
- Installation guide (docs/INSTALLATION.md)
- Training guide (docs/TRAINING.md)
- API reference (docs/API.md)
- Contributing guidelines (CONTRIBUTING.md)
- MIT License

#### Infrastructure
- Python package setup (setup.py)
- Requirements specification
- .gitignore configuration
- Project structure with modular organization

### Technical Details

**Python Version**: 3.8+

**Dependencies**:
- PyTorch >= 1.10.0
- ASE >= 3.22.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- matplotlib >= 3.3.0
- pandas >= 1.3.0
- tqdm >= 4.60.0

**Supported Platforms**:
- Linux
- macOS
- Windows

### Known Issues
- Graph data loading requires allow_pickle=True for npz files
- Normalization computation in Slater function needs to be dynamic to avoid gradient issues

### Contributors
- dreamchen (@junminchen)

---

## [Unreleased]

### Planned Features
- Additional force field components
- Multi-GPU training support
- Hyperparameter optimization utilities
- Visualization tools for trajectories
- Performance benchmarks
- Integration with DFT codes
- Pre-trained models for common electrolytes
- Jupyter notebook tutorials
- Unit tests
- CI/CD pipeline
