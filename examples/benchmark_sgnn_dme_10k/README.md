# DME 10k-Atom sGNN Benchmark

This folder contains isolated benchmarks for estimating the sGNN cost
of a 10,000-atom DME system, including a native OpenMM path.

## Layout

- `inputs/`: benchmark inputs copied from the repository
- `outputs/`: generated packed structure and timing reports
- `benchmark_dme_10k.py`: pure JAX sGNN benchmark
- `benchmark_openmm_native_dme_10k.py`: native OpenMM TorchForce benchmark

## Environment

Use the existing conda environment:

```bash
source /opt/anaconda3/bin/activate /opt/anaconda3/envs/openmm-ml-interface
```

## Run

```bash
cd examples/benchmark_sgnn_dme_10k
python benchmark_dme_10k.py
```

The native benchmark measures `OpenMM + TorchForce` stepping cost and reports
an estimated ns/day throughput.
