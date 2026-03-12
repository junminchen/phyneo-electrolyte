# MPIDForce + TorchForce CUDA Conflict: Troubleshooting Guide

## Problem

When combining `MPIDForce` (from mpidplugin) with `openmmtorch.TorchForce` on the
CUDA platform, `mm.Context(...)` fails with:

```
Exception: /path/to/libOpenMMTorchCUDA.so does not support all required kernels
```

Running `MPIDForce` alone on CUDA works fine; the error only appears when a
`TorchForce` is added to the same system.

## Root Cause

There are **two independent issues** that must both be fixed:

### 1. Missing torch shared libraries (environment issue)

`libOpenMMTorchCUDA.so` depends on `libtorch.so`, `libtorch_cuda.so`, etc.  In
some conda environments (e.g. the `mpid` env), these libraries live only under:

```
$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib/
```

but are **not** symlinked into the top-level `$CONDA_PREFIX/lib/` directory.  The
dynamic linker cannot find them at runtime, causing the "does not support all
required kernels" error.

In contrast, environments like `phyneo` have these symlinks already in place,
which is why the same code works there.

### 2. Non-differentiable TorchForce output (code issue)

If a `TorchForce` model returns a tensor with no `grad_fn` (e.g.
`torch.zeros(())`), openmmtorch's backward pass crashes on CUDA because it
cannot compute gradients with respect to positions.

The fix is to ensure the output always depends on the input positions, even for a
"zero energy" force:

```python
# BAD: no grad_fn, backward() will crash
return torch.zeros((), dtype=torch.float32)

# GOOD: differentiable w.r.t. positions
return torch.sum(positions) * 0.0
```

## Solution

### Fix 1: Symlink torch libraries (permanent, recommended)

Run this once per environment to create the missing symlinks:

```bash
CONDA_LIB=$CONDA_PREFIX/lib
TORCH_LIB=$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib

for lib in libtorch.so libtorch_cpu.so libtorch_cuda.so libc10.so libc10_cuda.so libtorch_python.so; do
    ln -sf "$TORCH_LIB/$lib" "$CONDA_LIB/$lib"
done
```

> **Note**: Replace `python3.11` with your actual Python version if different.

### Fix 2: Set LD_LIBRARY_PATH (temporary alternative)

If you prefer not to create symlinks, set `LD_LIBRARY_PATH` before running:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
python your_script.py --platform CUDA
```

This must be set **before** the Python process starts — setting it inside the
script after OpenMM libraries are already loaded may not help.

### Fix 3: Ensure TorchForce models are differentiable

Any `nn.Module` used with `openmmtorch.TorchForce` must return a scalar that is
differentiable w.r.t. the `positions` input.  This is required for the force
calculation (negative gradient of energy).

## Verification

After applying the fixes, both commands should succeed:

```bash
cd examples/md_simulation

# Without TorchForce (baseline)
python repro_mpid_openmmtorch_conflict.py --platform CUDA

# With TorchForce (the previously failing case)
python repro_mpid_openmmtorch_conflict.py --platform CUDA --with-torch-force
```

Expected output:

```
Context creation: OK
Potential energy: -367072.87535515684 kJ/mol
```

Both runs should report the same energy, confirming the zero-energy TorchForce
does not affect the MPIDForce result.

## Environment Details

| Component      | Version/Details                     |
|----------------|-------------------------------------|
| OpenMM         | 8.x with CUDA platform              |
| openmmtorch    | Compatible with PyTorch 2.x         |
| mpidplugin     | MPID force field plugin              |
| PyTorch        | 2.x with CUDA support               |
| conda env      | `mpid` (issue), `phyneo` (works)    |

## Key Takeaway

When setting up a new conda environment for MPIDForce + TorchForce:

1. Install PyTorch, OpenMM, openmmtorch, and mpidplugin
2. **Verify** that `libtorch.so` is accessible from `$CONDA_PREFIX/lib/` — if
   not, create symlinks as shown above
3. Ensure all TorchForce models produce differentiable outputs
