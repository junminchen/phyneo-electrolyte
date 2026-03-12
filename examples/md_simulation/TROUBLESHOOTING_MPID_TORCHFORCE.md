# MPIDForce + TorchForce / CallbackPyForce CUDA Conflict

## Problem

When combining `MPIDForce` (from mpidplugin) with either `openmmtorch.TorchForce`
or `CallbackPyForce.TorchForce` on the CUDA platform, `mm.Context(...)` fails:

```
Specified a Platform for a Context which does not support all required kernels
```

Running `MPIDForce` alone on CUDA works fine; the error only appears when a
Torch-based ML force is added to the same system.

## Root Cause

The error is **not** a kernel incompatibility between MPIDForce and TorchForce.
It is caused by **missing shared-library symlinks** in the conda environment.

### Why it happens

When PyTorch is pip-installed into a conda environment, its shared libraries are
split across two non-standard locations:

| Library group               | Installed under                                      |
|-----------------------------|------------------------------------------------------|
| `libtorch*.so`, `libc10*.so` | `$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib/` |
| `libcudnn.so.9`, `libcufile.so.0`, `libnccl.so.2`, `libcusparseLt.so.0` | `$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/*/lib/` |

Neither location is on the default library search path.  When OpenMM tries to
`dlopen` the CUDA kernel plugins (`libOpenMMTorchCUDA.so`,
`libCallbackPyForce_CUDA.so`), the dynamic linker cannot resolve these
dependencies, so the CUDA kernels fail to register.  OpenMM then reports the
misleading "does not support all required kernels" error.

In environments where PyTorch was installed via conda (e.g. the `phyneo` env),
these symlinks are created automatically, which is why the same code works there.

### Secondary issue: non-differentiable TorchForce output

For `openmmtorch.TorchForce` specifically, the model's `forward()` must return a
scalar that is differentiable w.r.t. positions.  A non-differentiable return
(e.g. `torch.zeros(())`) causes the backward pass to crash on CUDA:

```python
# BAD: no grad_fn, backward() will crash
return torch.zeros((), dtype=torch.float32)

# GOOD: differentiable w.r.t. positions
return torch.sum(positions) * 0.0
```

This does not apply to `CallbackPyForce`, which computes forces directly in the
callback rather than via autograd.

## Solution

### Fix 1: Symlink all missing libraries (permanent, recommended)

Run this once per environment:

```bash
CONDA_LIB="$CONDA_PREFIX/lib"
TORCH_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib"
NVIDIA_BASE="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia"

# Torch core libraries
for lib in libtorch.so libtorch_cpu.so libtorch_cuda.so \
           libc10.so libc10_cuda.so libtorch_python.so \
           libtorch_nvshmem.so libshm.so; do
    [ -f "$TORCH_LIB/$lib" ] && ln -sf "$TORCH_LIB/$lib" "$CONDA_LIB/$lib"
done

# NVIDIA runtime libraries
for pair in cudnn/lib/libcudnn.so.9 \
            cufile/lib/libcufile.so.0 \
            nccl/lib/libnccl.so.2 \
            cusparselt/lib/libcusparseLt.so.0; do
    src="$NVIDIA_BASE/$pair"
    [ -f "$src" ] && ln -sf "$src" "$CONDA_LIB/$(basename $pair)"
done
```

> **Note**: Replace `python3.11` with your actual Python version if different.

### Fix 2: Set LD_LIBRARY_PATH (temporary alternative)

If you prefer not to create symlinks, export both paths before running:

```bash
TORCH_LIB=$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib
NVIDIA_LIBS=$(python -c "
import pathlib, nvidia
base = pathlib.Path(nvidia.__file__).parent
print(':'.join(str(p) for p in base.glob('*/lib') if p.is_dir()))
")
export LD_LIBRARY_PATH=$TORCH_LIB:$NVIDIA_LIBS:$LD_LIBRARY_PATH
```

This must be set **before** the Python process starts — setting it inside the
script after shared libraries are already loaded will not help.

### Fix 3: Ensure TorchForce models are differentiable

Any `nn.Module` used with `openmmtorch.TorchForce` must return a scalar that is
differentiable w.r.t. the `positions` input.  This is required for the force
calculation (negative gradient of energy).

## Verification

After applying the fixes:

```bash
cd examples/md_simulation

# MPIDForce alone (baseline)
python repro_mpid_openmmtorch_conflict.py --platform CUDA

# MPIDForce + openmmtorch TorchForce
python repro_mpid_openmmtorch_conflict.py --platform CUDA --with-torch-force
```

Expected output:

```
Context creation: OK
Potential energy: -367072.87535515684 kJ/mol
```

Both runs should report the same energy, confirming the zero-energy TorchForce
does not affect the MPIDForce result.

## Diagnostic: Check for missing libraries

To check if your environment has unresolved shared-library dependencies:

```bash
# Check openmmtorch CUDA plugin
ldd $CONDA_PREFIX/lib/plugins/libOpenMMTorchCUDA.so | grep "not found"

# Check CallbackPyForce CUDA plugin (if installed)
ldd $CONDA_PREFIX/lib/plugins/libCallbackPyForce_CUDA.so | grep "not found"

# Check Python SWIG module
ldd $CONDA_PREFIX/lib/python3.11/site-packages/_CallbackPyForce.so | grep "not found"
```

If any lines appear, those libraries need to be symlinked as described above.

## Environment Details

| Component        | Version/Details                     |
|------------------|-------------------------------------|
| OpenMM           | 8.x with CUDA platform              |
| openmmtorch      | Compatible with PyTorch 2.x         |
| CallbackPyForce  | 0.0.0 (SWIG-based OpenMM plugin)    |
| mpidplugin       | MPID force field plugin              |
| PyTorch          | 2.x with CUDA support (pip install) |
| conda env        | `mpid` (issue), `phyneo` (works)    |

## Key Takeaway

When setting up a new conda environment for MPIDForce + TorchForce/CallbackPyForce:

1. Install PyTorch, OpenMM, openmmtorch/CallbackPyForce, and mpidplugin
2. **Symlink** all torch and NVIDIA runtime libraries into `$CONDA_PREFIX/lib/`
3. Verify with `ldd` that no plugin `.so` has "not found" dependencies
4. For openmmtorch: ensure all TorchForce models produce differentiable outputs
