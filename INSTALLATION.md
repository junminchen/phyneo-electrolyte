# Installation Guide

This guide covers setting up a working environment for PhyNEO-Electrolyte,
including GPU-accelerated MD simulations with MPIDForce + TorchForce.

## Prerequisites

- Linux (tested on Ubuntu)
- NVIDIA GPU with CUDA support (for GPU-accelerated simulations)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or
  [Mamba](https://mamba.readthedocs.io/)
- Git LFS (`git lfs install`)

## Quick Install

```bash
# Clone with Git LFS
git clone https://github.com/junminchen/phyneo-electrolyte.git
cd phyneo-electrolyte
git lfs pull

# Install dependencies and package
pip install -r requirements.txt
pip install -e .
```

This works if you already have a suitable conda environment.  For a fresh setup,
follow the detailed instructions below.

---

## Detailed Setup: Conda Environment from Scratch

### Step 1: Create the conda environment

```bash
conda create -n phyneo python=3.11 -y
conda activate phyneo
```

### Step 2: Install CUDA toolkit (conda-managed)

```bash
conda install -c nvidia cuda-toolkit=12.8 -y
```

Or match the CUDA version your GPU driver supports (`nvidia-smi` to check).

### Step 3: Install OpenMM

```bash
conda install -c conda-forge openmm=8.2.0 -y
```

Verify:

```bash
python -c "import openmm; print(openmm.__version__, openmm.Platform.getNumPlatforms())"
```

### Step 4: Install PyTorch (with CUDA)

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Adjust `cu124` to match your CUDA version (e.g. `cu118`, `cu121`).

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Step 5: Install JAX (with CUDA)

```bash
pip install jax==0.4.26 jaxlib==0.4.26+cuda12.cudnn91 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Verify:

```bash
python -c "import jax; print(jax.__version__, jax.devices())"
```

### Step 6: Install OpenMM-Torch plugin

```bash
conda install -c conda-forge openmm-torch>=1.6.0 -y
```

Or via pip:

```bash
pip install openmm-torch>=1.6.0
```

### Step 7: Install DMFF (custom branch)

```bash
pip install "dmff @ git+https://github.com/junminchen/DMFF.git@devel"
```

### Step 8: Install remaining dependencies and the package

```bash
cd phyneo-electrolyte
pip install -r requirements.txt
pip install -e .
```

### Step 9 (Optional): Install MPID plugin

If you need `MPIDForce` for long-range electrostatics (required for production
MD), install `mpidplugin` into the same environment.  This is typically done via
conda from the plugin's own channel or by building from source.

```bash
conda install -c conda-forge mpidplugin -y
```

---

## Post-Install: Symlink Torch Libraries

**This step is critical** when using `openmmtorch` TorchForce on CUDA.

In some conda environments, `libtorch*.so` libraries are installed only under
`site-packages/torch/lib/` and are **not** symlinked into the environment's
top-level `lib/` directory.  This causes OpenMM to fail with:

```
libOpenMMTorchCUDA.so does not support all required kernels
```

### Fix: Create symlinks

```bash
CONDA_LIB=$CONDA_PREFIX/lib
TORCH_LIB=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")

for lib in libtorch.so libtorch_cpu.so libtorch_cuda.so libc10.so libc10_cuda.so libtorch_python.so; do
    [ -f "$TORCH_LIB/$lib" ] && ln -sf "$TORCH_LIB/$lib" "$CONDA_LIB/$lib"
done
```

### Verify the fix

```bash
python -c "
import openmm as mm
print('Platforms:', [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())])
import openmmtorch
print('openmmtorch: OK')
"
```

### Alternative: Set LD_LIBRARY_PATH

If you prefer not to create symlinks, export the path before running scripts:

```bash
export LD_LIBRARY_PATH=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')"):$LD_LIBRARY_PATH
```

Add this line to your `~/.bashrc` or conda `activate.d/` script for persistence.

---

## Verification

### Basic import test

```bash
python -c "
import phyneo
import openmm
import torch
import jax
print('phyneo:', phyneo.__version__)
print('openmm:', openmm.__version__)
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('jax:', jax.__version__, '| devices:', jax.devices())
print('All imports OK')
"
```

### MPIDForce + TorchForce integration test

```bash
cd examples/md_simulation

# MPIDForce alone (baseline)
python repro_mpid_openmmtorch_conflict.py --platform CUDA

# MPIDForce + TorchForce (requires symlink fix above)
python repro_mpid_openmmtorch_conflict.py --platform CUDA --with-torch-force
```

Both should print:

```
Context creation: OK
Potential energy: -367072.87535515684 kJ/mol
```

### Run tests

```bash
pytest tests/
```

---

## Dependency Summary

| Package         | Version   | Source           | Purpose                          |
|-----------------|-----------|------------------|----------------------------------|
| Python          | >=3.10    | conda            | Runtime                          |
| PyTorch         | 2.6.0     | pip (CUDA wheel) | ML models, TorchForce            |
| JAX             | 0.4.26    | pip (CUDA wheel) | Training, DMFF backend           |
| jaxlib          | 0.4.26    | pip (CUDA wheel) | JAX backend                      |
| OpenMM          | 8.2.0     | conda-forge      | MD engine                        |
| openmm-torch    | >=1.6.0   | conda-forge/pip  | PyTorch models in OpenMM         |
| DMFF            | devel     | pip (git)        | Differentiable force field       |
| flax            | 0.8.4     | pip              | Neural network library (JAX)     |
| optax           | 0.2.2     | pip              | Optimizer (JAX)                  |
| numpy           | 1.26.0    | pip              | Array computation                |
| scipy           | 1.10.1    | pip              | Scientific computing             |
| ase             | 3.25.0    | pip              | Atomic simulation environment    |
| rdkit           | 2023.9.6  | pip              | Cheminformatics                  |
| mpidplugin      | >=1.0     | conda-forge      | MPID long-range electrostatics   |

---

## Known Issues

### MPIDForce + TorchForce CUDA conflict

See [examples/md_simulation/TROUBLESHOOTING_MPID_TORCHFORCE.md](examples/md_simulation/TROUBLESHOOTING_MPID_TORCHFORCE.md)
for detailed root cause analysis and fixes.

### JAX/PyTorch CUDA version mismatch

JAX and PyTorch must be installed with matching CUDA versions.  A mismatch
(e.g. JAX on CUDA 12 + PyTorch on CUDA 11) causes silent failures or crashes.
Check with:

```bash
python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"
python -c "import jax; print('JAX devices:', jax.devices())"
```

### DMFF optional warnings

DMFF may print warnings about missing RDKit, dpdpnblist, or Parmed at import
time.  These are safe to ignore unless you need those specific features.

---

## Development Setup

```bash
pip install -e ".[dev]"

# Format
ruff format phyneo

# Lint
ruff check phyneo

# Type check
mypy
```
