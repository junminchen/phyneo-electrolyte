# MPID + TorchForce Report

Date: March 13, 2026

## Scope

This report checks three runtime paths in `examples/md_simulation/`:

- `MPIDForce` alone
- `MPIDForce + sGNNForceFast` via `openmmtorch`
- `MPIDForce + tiled fixed-topology sGNNForceFast` via `openmmtorch`
- `MPIDForce + sGNNForceFast` via `CallbackPyForce`

## Final Status

### 1. MPID-only

Status: works

Benchmark command:

```bash
/home/am3-peichenzhong-group/miniconda3/envs/mpid/bin/python -u - <<'PY'
import time
import openmm as mm
from openmm import app, unit
import mpidplugin

pdb = app.PDBFile('examples/md_simulation/init.pdb')
ff = app.ForceField('examples/md_simulation/phyneo_ecl.xml')
system = ff.createSystem(
    pdb.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=0.6*unit.nanometer,
    constraints=None,
    rigidWater=False,
    removeCMMotion=True,
)
integrator = mm.LangevinIntegrator(298*unit.kelvin, 1.0/unit.picosecond, 1.0*unit.femtosecond)
platform = mm.Platform.getPlatformByName('CUDA')
sim = app.Simulation(pdb.topology, system, integrator, platform, {'CudaPrecision':'mixed'})
sim.context.setPositions(pdb.positions)
sim.step(10)
t0=time.perf_counter(); sim.step(100); dt=time.perf_counter()-t0
print(f'MPID_ONLY_TIME {dt:.6f}')
print(f'MPID_ONLY_MS_PER_STEP {dt/100*1000:.6f}')
print(f'MPID_ONLY_NS_PER_DAY {100*1e-6/dt*86400:.6f}')
PY
```

Measured performance:

- `1.056 ms/step`
- `81.786 ns/day`

### 2. MPID + sGNN via openmmtorch

Status: works

Script:

- `examples/md_simulation/openmm_mpid_sgnn_openmmtorch.py`

Important implementation note:

- the script now auto-adds the needed Torch/OpenMM runtime libraries to
  `LD_LIBRARY_PATH`
- it runs directly without needing a manual wrapper command

Benchmark command:

```bash
/home/am3-peichenzhong-group/miniconda3/envs/mpid/bin/python -u \
  examples/md_simulation/openmm_mpid_sgnn_openmmtorch.py \
  --steps 100 --warmup 10 --report-interval 0 --benchmark
```

Measured performance:

- `25.961 ms/step`
- `3.328 ns/day`

Energy breakdown at initialization:

- total: `311775.8745 kJ/mol`
- MPID: `-367072.8759 kJ/mol`
- sGNN: `678848.75 kJ/mol`

This path also completed a `1`-step MD sanity check successfully.

### 3. MPID + sGNN via CallbackPyForce

Status: works after adding the required NVSHMEM/Torch runtime library paths

Script:

- `examples/md_simulation/openmm_mpid_sgnn_fast.py`

Runtime note:

- this path currently requires the following directories on `LD_LIBRARY_PATH`
  before startup:
  - `.../site-packages/torch/lib`
  - `.../site-packages/nvidia/cufile/lib`
  - `.../site-packages/nvidia/nvshmem/lib`

Benchmark command:

```bash
/home/am3-peichenzhong-group/miniconda3/envs/mpid/bin/python -u \
  examples/md_simulation/openmm_mpid_sgnn_fast.py \
  --steps 100 --warmup 10 --report-interval 0 --benchmark
```

Equivalent tested invocation:

```bash
LD_LIBRARY_PATH=/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/python3.11/site-packages/torch/lib:\
/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/python3.11/site-packages/nvidia/cufile/lib:\
/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/python3.11/site-packages/nvidia/nvshmem/lib:$LD_LIBRARY_PATH \
/home/am3-peichenzhong-group/miniconda3/envs/mpid/bin/python -u \
  examples/md_simulation/openmm_mpid_sgnn_fast.py \
  --steps 100 --warmup 10 --report-interval 0 --benchmark
```

Measured performance:

- `27.649 ms/step`
- `3.125 ns/day`

Energy breakdown at initialization:

- total: `311775.8747 kJ/mol`
- MPID: `-367072.8759 kJ/mol`
- sGNN: `678848.75 kJ/mol`

### 4. MPID + tiled fixed-topology sGNN via openmmtorch

Status: works and is much faster than the untiled `openmmtorch` path

Script:

- `examples/md_simulation/openmm_mpid_sgnn_openmmtorch_tiled.py`

Approach:

- pre-expand the fixed residue topology before MD
- batch repeated molecules by residue type
  - `DMC x140`
  - `ECA x45`
  - `PF6 x17`
- avoid the per-step Python-side `index_select` loops used in the baseline
  `openmmtorch` script

Benchmark command:

```bash
/home/am3-peichenzhong-group/miniconda3/envs/mpid/bin/python -u \
  examples/md_simulation/openmm_mpid_sgnn_openmmtorch_tiled.py \
  --steps 100 --warmup 10 --report-interval 0 --benchmark
```

Measured performance:

- `5.725 ms/step`
- `15.092 ns/day`

Energy breakdown at initialization:

- total: `311775.6870 kJ/mol`
- MPID: `-367072.8757 kJ/mol`
- sGNN: `678848.5625 kJ/mol`

Compared with the untiled `openmmtorch` path:

- baseline `openmmtorch`: `25.961 ms/step`
- `CallbackPyForce`: `27.649 ms/step`
- tiled `openmmtorch`: `5.725 ms/step`
- tiled speedup vs baseline `openmmtorch`: about `4.5x`
- tiled speedup vs `CallbackPyForce`: about `4.8x`

## Minimal Reproducer

Minimal reproducer script:

- `examples/md_simulation/repro_mpid_openmmtorch_conflict.py`

Current result:

- `MPIDForce` alone on `CUDA`: works
- `MPIDForce + trivial zero-energy openmmtorch force` on `CUDA`: works

So the original generic TorchForce conflict has been fixed.

## Interpretation

The remaining issue is narrower:

- generic `MPIDForce + TorchForce` now works
- the repository's actual `MPIDForce + openmmtorch sGNN` path works
- the tiled fixed-topology `MPIDForce + openmmtorch sGNN` path works and is much faster
- the `CallbackPyForce` route also works once its dependent runtime libraries are visible
- however, `CallbackPyForce` is still slower than the current `openmmtorch` path
- and much slower than the tiled fixed-topology `openmmtorch` path

That means the current blocker is now specifically the `CallbackPyForce`
integration path, not TorchForce in general.

## Files

- `examples/md_simulation/openmm_mpid_sgnn_openmmtorch.py`
- `examples/md_simulation/openmm_mpid_sgnn_openmmtorch_tiled.py`
- `examples/md_simulation/openmm_mpid_sgnn_fast.py`
- `examples/md_simulation/repro_mpid_openmmtorch_conflict.py`
- `OPENMM_MPID_SGNN_DEBUG.md`
