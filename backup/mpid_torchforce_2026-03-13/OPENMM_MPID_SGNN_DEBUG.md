# OpenMM MPID + sGNN Integration Notes

Date: March 13, 2026

## Summary

The `MPIDForce` and `sGNNForceFast` pieces both work on their own in the `mpid`
environment, but neither `CallbackPyForce` nor `openmmtorch` currently works
when the ML force is combined with `MPIDForce` inside the same OpenMM `System`.

So the current blocker is broader than the callback bridge. The problem appears
to be the combined `MPIDForce + Torch-based ML force` kernel stack.

## What Was Tested

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

Status: fails

Observed result:

- Building a single combined `System` and then constructing a `Context` fails on
  OpenMM platform `CUDA` with:
  `Specified a Platform for a Context which does not support all required kernels`
- Trying the same combined script on OpenMM platform `CPU` also failed with the
  same kernel-support error.

### 4. Create MPID-only `Context`, then `system.addForce(callback_force)`, then `reinitialize()`

Status: fails

Observed result:

- MPID-only `CUDA Context` is created successfully first.
- After adding the callback force and calling `context.reinitialize(preserveState=True)`,
  the first energy evaluation fails with:
  `CUDA error: an illegal memory access was encountered`
- The Python stack lands in
  `examples/torch_gnn/sgnn_fast.py` during `calc_internal_coords_features()`.

## Interpretation

The failure is not caused by:

- MPID alone
- `CallbackPyForce` alone
- the order of `createSystem()` versus `system.addForce()`

The failure only appears when the MPID plugin and `CallbackPyForce` are combined.

That strongly suggests a kernel/runtime incompatibility between:

- the MPID plugin's OpenMM kernels
- and the `CallbackPyForce` execution path

on this environment and platform stack.

## Additional Implementation Note

To support ABn species such as `PF6`, the Torch/OpenMM runtime cannot rely on
`examples/torch_gnn/graph_torch.py` alone, because that path still hard-codes
`MAX_VALENCE = 4`.

The current runtime script therefore switched graph construction to the training
side implementation in `dmff.sgnn.graph`, which already supports
`max_valence=6`.

### 5. `MPIDForce + openmmtorch TorchForce` in the same `System`

Status: fails

Observed result:

- `CUDA`: the combined system fails during `Simulation(...)` / `Context(...)`
  creation with:
  `Specified a Platform for a Context which does not support all required kernels`
- `CPU`: the same combined system fails with the same error

This means the issue is not specific to `CallbackPyForce`. It also affects the
`openmmtorch` integration path.

## Current Interpretation

The failure is not caused by:

- MPID alone
- `CallbackPyForce` alone
- `openmmtorch` alone on non-MPID systems
- the order of `createSystem()` versus `system.addForce()`

The failure only appears when the MPID plugin and a Torch-based ML force are
combined in one OpenMM `System`.

The most likely interpretation is:

- the MPID plugin and TorchForce-style ML plugins are exposing incompatible
  kernel requirements to OpenMM on this environment
- or the current MPID plugin build does not advertise support for the same
  platform/kernel combinations required by the ML force plugins

## Recommended Path Forward

Short term:

- keep `MPIDForce` and sGNN outside the same native OpenMM `System`
- use the existing `client_dmff.py`-style external/JAX composition path for now

Medium term:

- inspect the MPID plugin build and OpenMM plugin registration in the `mpid`
  environment
- check whether a different OpenMM / plugin build matrix is needed for
  `MPIDForce + TorchForce`
- test a minimal standalone `MPIDForce + trivial openmmtorch model` reproducer
  outside this repository to isolate whether the incompatibility is generic

## Relevant Files

- `examples/md_simulation/repro_mpid_openmmtorch_conflict.py`
- `examples/md_simulation/openmm_mpid_sgnn_fast.py`
- `examples/md_simulation/openmm_mpid_sgnn_openmmtorch.py`
- `examples/torch_gnn/sgnn_fast.py`
- `examples/md_simulation/phyneo_ecl.xml`
- `examples/torch_gnn/bench_openmm.py`
