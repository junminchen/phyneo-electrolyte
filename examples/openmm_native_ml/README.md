# OpenMM Native ML Integration Examples

This directory contains reference implementations for integrating PhyNEO Machine Learning potentials (**EAPNN** and **sGNN**) directly into OpenMM. By using the `openmm-torch` plugin, these models run natively within the OpenMM C++ engine, providing significant performance gains over socket-based communication.

## Example Scripts

### 🚀 Minimal Implementations
These scripts provide the simplest starting point for a single-point energy calculation:
- **`minimal_eapnn.py`**: Integration of the Pairwise Machine Learning Potential correction.
- **`minimal_sgnn.py`**: Integration of the Slater-based Graph Neural Network force.

### 🧪 Validation & Regression
These scripts verify that the PyTorch/OpenMM implementation matches the original JAX-DMFF research code:
- **`compare_jax_torch_eapnn.py`**: Numerical equivalency test for EAPNN.
- **`compare_jax_torch_sgnn.py`**: Numerical equivalency test for sGNN.

## Prerequisites

- `openmm`
- `pytorch`
- `openmm-torch`

You can install the requirements via conda:
```bash
conda install -c conda-forge openmm-torch
```

## Usage

Run any script from the project root:
```bash
# From repository root
python examples/openmm_native_ml/minimal_eapnn.py
```
