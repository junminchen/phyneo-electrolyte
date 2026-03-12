# MPID + TorchForce Report

Date: March 13, 2026

## Scope

This report checks three runtime paths in `examples/md_simulation/`:

- `MPIDForce` alone
- `MPIDForce + sGNNForceFast` via `openmmtorch`
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

Status: still fails

Script:

- `examples/md_simulation/openmm_mpid_sgnn_fast.py`

Test command:

```bash
/home/am3-peichenzhong-group/miniconda3/envs/mpid/bin/python -u \
  examples/md_simulation/openmm_mpid_sgnn_fast.py \
  --steps 0 --warmup 0 --report-interval 0
```

Observed failure:

```text
openmm.OpenMMException: Specified a Platform for a Context which does not support all required kernels
```

Because the combined system never reaches a valid `Context`, there is no
meaningful production-speed number for this path yet.

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
- but the `CallbackPyForce` route still does not work with `MPIDForce`

That means the current blocker is now specifically the `CallbackPyForce`
integration path, not TorchForce in general.

## Files

- `examples/md_simulation/openmm_mpid_sgnn_openmmtorch.py`
- `examples/md_simulation/openmm_mpid_sgnn_fast.py`
- `examples/md_simulation/repro_mpid_openmmtorch_conflict.py`
- `OPENMM_MPID_SGNN_DEBUG.md`
