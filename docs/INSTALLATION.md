# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/junminchen/phyneo-electrolyte.git
cd phyneo-electrolyte
```

### 2. Create Virtual Environment (Recommended)

Using venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Using conda:
```bash
conda create -n phyneo python=3.8
conda activate phyneo
```

### 3. Install Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

### 4. Install PhyNEO-Electrolyte

Development mode (recommended for contributing):
```bash
pip install -e .
```

Standard installation:
```bash
pip install .
```

### 5. Verify Installation

Test the installation:
```bash
python -c "import phyneo; print(phyneo.__version__)"
```

You should see the version number printed.

## GPU Support

### CUDA Installation

For GPU acceleration, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Optional Dependencies

### Development Tools

For development and testing:
```bash
pip install -e ".[dev]"
```

This installs:
- pytest: For running tests
- black: Code formatter
- flake8: Linter

## Troubleshooting

### Issue: Import Error

If you get import errors, ensure you're in the correct environment:
```bash
which python  # Should point to your virtual environment
pip list | grep phyneo
```

### Issue: PyTorch CUDA Not Available

Check CUDA installation:
```bash
nvidia-smi  # Should show GPU information
nvcc --version  # Should show CUDA compiler version
```

Reinstall PyTorch with correct CUDA version.

### Issue: ASE Installation Problems

Try installing ASE separately:
```bash
pip install --upgrade ase
```

## System-Specific Instructions

### Linux

Standard installation should work. Ensure build tools are installed:
```bash
sudo apt-get install build-essential python3-dev
```

### macOS

Install Xcode command line tools:
```bash
xcode-select --install
```

### Windows

Install Visual C++ Build Tools if needed for compiling extensions.

## Next Steps

After installation:
1. Check the [Quick Start Guide](../README.md#quick-start)
2. Run the [MD Example](../examples/md_simulation/README.md)
3. Prepare your training data
4. Start training models

For more details, see the full documentation.
