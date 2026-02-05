# MD Simulation Example

This directory contains an example of running molecular dynamics simulations
using the PhyNEO-Electrolyte force field.

## Overview

The example demonstrates:
- Creating a simple liquid electrolyte system
- Setting up a PhyNEO calculator
- Running MD simulations with ASE
- Analyzing trajectory data

## Files

- `run_md_example.py`: Main script for running MD simulation
- `README.md`: This file

## Usage

Run the MD simulation example:

```bash
cd examples/md_simulation
python run_md_example.py
```

This will:
1. Create a simple electrolyte system with Li+ ions and PF6- ions
2. Run a short MD simulation at 300 K
3. Save trajectory to `trajectory.traj`
4. Save initial and final structures as XYZ files

## Requirements

- Python 3.8+
- PyTorch
- ASE (Atomic Simulation Environment)
- NumPy

Install requirements:
```bash
pip install -r ../../requirements.txt
```

## Expected Output

The script will generate:
- `initial_structure.xyz`: Initial atomic configuration
- `final_structure.xyz`: Final atomic configuration after MD
- `trajectory.traj`: Full MD trajectory (ASE format)

Console output will show:
- Energy values at each logged step
- Temperature evolution
- Simulation statistics

## Customization

You can modify the simulation parameters in `run_md_example.py`:
- `temperature`: Target temperature in Kelvin
- `timestep`: MD timestep in femtoseconds
- `nsteps`: Number of MD steps
- `friction`: Friction coefficient for Langevin dynamics

## Advanced Usage

For production simulations:
1. Load trained PhyNEO models (Slater, Pairwise, Subgraph)
2. Implement full calculator with all model components
3. Use longer simulation times
4. Add analysis tools (RDF, MSD, etc.)

See the main documentation for more details.
