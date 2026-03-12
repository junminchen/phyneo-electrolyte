# OpenMM MPID + sGNN Integration Notes

Note: this document is primarily a historical debug log.  The root cause and fix
are summarized in
`examples/md_simulation/TROUBLESHOOTING_MPID_TORCHFORCE.md`.

Date: March 13, 2026

## Resolution (2026-03-13)

All previously reported failures — `MPIDForce + openmmtorch TorchForce` **and**
`MPIDForce + CallbackPyForce` — have been traced to a single root cause:
**missing shared-library symlinks** in the `mpid` conda environment.

When PyTorch is pip-installed, its CUDA libraries (`libtorch*.so`, `libc10*.so`)
land under `site-packages/torch/lib/`, and NVIDIA runtime libraries
(`libcudnn.so.9`, `libcufile.so.0`, `libnccl.so.2`, `libcusparseLt.so.0`) land
under `site-packages/nvidia/*/lib/`.  Neither set is symlinked into
`$CONDA_PREFIX/lib/`, so OpenMM's plugin loader cannot find them when it tries to
`dlopen` the CUDA kernel libraries (`libOpenMMTorchCUDA.so`,
`libCallbackPyForce_CUDA.so`).  The result is the misleading error:

```
Specified a Platform for a Context which does not support all required kernels
```

The fix is a one-time symlink step (see Troubleshooting doc).

Current status after fix:

- `MPIDForce + openmmtorch TorchForce` on CUDA: **works**
- `MPIDForce + CallbackPyForce TorchForce` on CUDA: **works**
- Energy: `-367072.88 kJ/mol` (consistent across all paths)

---

## Historical Debug Log

The sections below record the original investigation before the root cause was
identified.  They are kept for context.

### 1. `CallbackPyForce + sGNNForceFast` by itself

Status: works

Observed result:

- `CUDA`: context creation succeeded
- `CPU`: context creation succeeded
- `Reference`: context creation succeeded
- Reported energy for `examples/md_simulation/init.pdb`:
  `678848.75 kJ/mol`

This means the Torch-side sGNN model and the `CallbackPyForce` wrapper are not
broken on their own.

### 2. `MPIDForce` by itself

Status: works

Observed result:

- `CUDA`: context creation succeeded
- `CPU`: context creation succeeded
- `Reference`: context creation succeeded

Observed energies:

- `CUDA`: about `-367072.88 kJ/mol`
- `CPU`/`Reference`: about `-46374.97 kJ/mol`

This confirms the MPID plugin is loading correctly from
`examples/md_simulation/phyneo_ecl.xml` and can run independently.

### 3. `MPIDForce + CallbackPyForce` in the same `System`

Status: originally failed, now **fixed** (symlink issue)

Original error:

- `Specified a Platform for a Context which does not support all required kernels`
- Occurred on both CUDA and CPU platforms

Root cause: `libCallbackPyForce_CUDA.so` links against `libtorch_cuda.so`, which
transitively depends on `libcudnn.so.9`, `libcufile.so.0`, `libnccl.so.2`, and
`libcusparseLt.so.0`.  These libraries were installed by pip under
`site-packages/nvidia/*/lib/` but not symlinked into `$CONDA_PREFIX/lib/`.

### 4. Create MPID-only `Context`, then `system.addForce(callback_force)`, then `reinitialize()`

Status: originally failed with `CUDA error: an illegal memory access`

This was a secondary symptom of the same missing-library issue.  The CUDA plugin
was partially loaded, leading to memory corruption during force evaluation.

### 5. `MPIDForce + openmmtorch TorchForce` in the same `System`

Status: originally failed, now **fixed** (symlink + differentiable output)

Two fixes were needed:

1. Symlink `libtorch*.so` and `libc10*.so` into `$CONDA_PREFIX/lib/`
2. Ensure TorchForce model output is differentiable w.r.t. positions
   (`torch.sum(positions) * 0.0` instead of `torch.zeros(())`)

## Additional Implementation Note

To support ABn species such as `PF6`, the Torch/OpenMM runtime cannot rely on
`examples/torch_gnn/graph_torch.py` alone, because that path still hard-codes
`MAX_VALENCE = 4`.

The current runtime script therefore switched graph construction to the training
side implementation in `dmff.sgnn.graph`, which already supports
`max_valence=6`.

## Relevant Files

- `examples/md_simulation/repro_mpid_openmmtorch_conflict.py`
- `examples/md_simulation/TROUBLESHOOTING_MPID_TORCHFORCE.md`
- `examples/md_simulation/openmm_mpid_sgnn_fast.py`
- `examples/md_simulation/openmm_mpid_sgnn_openmmtorch.py`
- `examples/torch_gnn/sgnn_fast.py`
- `examples/md_simulation/phyneo_ecl.xml`
- `examples/torch_gnn/bench_openmm.py`
