#!/usr/bin/env python

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmmtorch
import torch
import torch.nn as nn
from openmm import app, unit

from dmff.sgnn.graph import from_pdb
from dmff.sgnn.gnn import MolGNNForce
from phyneo.models.torch_models import sGNNForceTorch


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
TARGET_ATOMS = 10_000
GRID_SPACING_ANGSTROM = 9.0
TIMESTEP_FS = 1.0
WARMUP_STEPS = 5
MEASURE_BLOCKS = 5
STEPS_PER_BLOCK = 20


def load_monomer():
    pdb = app.PDBFile(str(INPUTS / "DME.pdb"))
    positions_angstrom = np.asarray(pdb.positions._value, dtype=np.float64) * 10.0
    atoms = [(atom.name, atom.element.symbol if atom.element else "X") for atom in pdb.topology.atoms()]
    bonds = [(atom1.index, atom2.index) for atom1, atom2 in pdb.topology.bonds()]
    return positions_angstrom, atoms, bonds


def build_offsets(molecule_count: int, spacing_angstrom: float) -> tuple[list[np.ndarray], float]:
    grid_side = math.ceil(molecule_count ** (1.0 / 3.0))
    offsets = []
    for ix in range(grid_side):
        for iy in range(grid_side):
            for iz in range(grid_side):
                offsets.append(np.array([ix, iy, iz], dtype=np.float64) * spacing_angstrom)
                if len(offsets) == molecule_count:
                    box_length = grid_side * spacing_angstrom + spacing_angstrom
                    return offsets, box_length
    raise RuntimeError("failed to build enough offsets")


def build_replicated_positions() -> tuple[np.ndarray, int, int, float, list[tuple[str, str]], list[tuple[int, int]]]:
    positions_angstrom, atoms, bonds = load_monomer()
    atoms_per_molecule = len(atoms)
    if TARGET_ATOMS % atoms_per_molecule != 0:
        raise ValueError("TARGET_ATOMS must be divisible by atoms per DME molecule")

    molecule_count = TARGET_ATOMS // atoms_per_molecule
    offsets, box_length = build_offsets(molecule_count, GRID_SPACING_ANGSTROM)

    all_positions = np.concatenate(
        [positions_angstrom + offset for offset in offsets],
        axis=0,
    )
    return all_positions, atoms_per_molecule, molecule_count, box_length, atoms, bonds


def write_large_pdb(output_path: Path) -> tuple[int, int, float, np.ndarray]:
    all_positions, atoms_per_molecule, molecule_count, box_length, atoms, bonds = build_replicated_positions()

    if output_path.exists():
        return atoms_per_molecule, molecule_count, box_length, all_positions

    with output_path.open("w", encoding="ascii") as handle:
        handle.write("HEADER    DME 10K OPENMM NATIVE BENCHMARK\n")
        handle.write(
            "CRYST1"
            f"{box_length:9.3f}{box_length:9.3f}{box_length:9.3f}"
            "  90.00  90.00  90.00 P 1           1\n"
        )

        serial = 1
        for residue_id in range(1, molecule_count + 1):
            start = (residue_id - 1) * atoms_per_molecule
            stop = start + atoms_per_molecule
            for (atom_name, element), position in zip(atoms, all_positions[start:stop]):
                handle.write(
                    f"ATOM  {serial:5d} {atom_name:>4s} DME A{residue_id:4d}"
                    f"    {position[0]:8.3f}{position[1]:8.3f}{position[2]:8.3f}"
                    f"  1.00  0.00          {element:>2s}\n"
                )
                serial += 1

        for molecule_index in range(molecule_count):
            base = molecule_index * atoms_per_molecule + 1
            for atom_i, atom_j in bonds:
                handle.write(f"CONECT{base + atom_i:5d}{base + atom_j:5d}\n")
        handle.write("END\n")

    return atoms_per_molecule, molecule_count, box_length, all_positions


def load_sgnn_params(model: sGNNForceTorch, pickle_path: Path) -> str:
    import pickle

    with pickle_path.open("rb") as handle:
        jax_params = pickle.load(handle)
    if "params" in jax_params:
        jax_params = jax_params["params"]

    state_dict = model.state_dict()
    if "w" in jax_params:
        state_dict["w"] = torch.tensor(np.array(jax_params["w"]), dtype=torch.float32).reshape(1)
    for module_name, js_name in [("fc0", "fc0"), ("fc1", "fc1")]:
        if f"{js_name}.weight" in jax_params:
            for index, (weight, bias) in enumerate(
                zip(jax_params[f"{js_name}.weight"], jax_params[f"{js_name}.bias"])
            ):
                state_dict[f"{module_name}.{index}.weight"] = torch.tensor(
                    np.array(weight), dtype=torch.float32
                )
                state_dict[f"{module_name}.{index}.bias"] = torch.tensor(
                    np.array(bias), dtype=torch.float32
                )
    if "fc_final.weight" in jax_params:
        state_dict["fc_final.weight"] = torch.tensor(
            np.array(jax_params["fc_final.weight"]), dtype=torch.float32
        )
        state_dict["fc_final.bias"] = torch.tensor(
            np.array(jax_params["fc_final.bias"]), dtype=torch.float32
        ).reshape(1)
    try:
        model.load_state_dict(state_dict)
        return "loaded"
    except RuntimeError:
        return "model_init"


class BatchedDMEForceWrapper(nn.Module):
    def __init__(self, core_model: sGNNForceTorch, atoms_per_molecule: int, molecule_count: int):
        super().__init__()
        self.core_model = core_model
        self.atoms_per_molecule = atoms_per_molecule
        self.molecule_count = molecule_count
        self.nm_to_angstrom = 10.0
        self.kcal_to_kj = 4.184
    def forward(self, positions: torch.Tensor, boxvectors: torch.Tensor) -> torch.Tensor:
        pos_angstrom = positions.float() * self.nm_to_angstrom
        box_angstrom = boxvectors.float() * self.nm_to_angstrom
        total_energy = torch.zeros(1, dtype=pos_angstrom.dtype, device=pos_angstrom.device)
        for molecule_index in range(self.molecule_count):
            start = molecule_index * self.atoms_per_molecule
            stop = start + self.atoms_per_molecule
            total_energy = total_energy + self.core_model(pos_angstrom[start:stop], box_angstrom)
        return total_energy.squeeze(0) * self.kcal_to_kj


def create_scripted_torch_force(
    system: mm.System,
    wrapped_model: nn.Module,
    cache_path: Path,
    force_group: int = 1,
) -> mm.System:
    if not cache_path.exists():
        scripted_model = torch.jit.script(wrapped_model)
        scripted_model.save(str(cache_path))
    torch_force = openmmtorch.TorchForce(str(cache_path))
    torch_force.setUsesPeriodicBoundaryConditions(True)
    torch_force.setForceGroup(force_group)
    system.addForce(torch_force)
    return system


def ns_per_day(step_time_seconds: float, timestep_fs: float) -> float:
    return 86_400.0 * timestep_fs / (step_time_seconds * 1.0e6)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    large_pdb = OUTPUTS / "dme_10000_atoms_native_openmm.pdb"
    cached_torchscript = OUTPUTS / "dme_10000_atoms_native_openmm.pt"
    atoms_per_molecule, molecule_count, box_length, all_positions_angstrom = write_large_pdb(large_pdb)

    monomer_graph = from_pdb(str(INPUTS / "DME.pdb"))
    _ = MolGNNForce(monomer_graph, nn=1)
    model = sGNNForceTorch(monomer_graph, n_layers=(3, 2), sizes=[(40, 20, 20), (20, 10)])
    param_source = load_sgnn_params(model, INPUTS / "params_sgnn.pickle")
    model.eval()

    system = mm.System()
    for _ in range(TARGET_ATOMS):
        system.addParticle(1.0 * unit.amu)
    box_nm = box_length / 10.0
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_nm, 0.0, 0.0) * unit.nanometer,
        mm.Vec3(0.0, box_nm, 0.0) * unit.nanometer,
        mm.Vec3(0.0, 0.0, box_nm) * unit.nanometer,
    )

    wrapped_model = BatchedDMEForceWrapper(model, atoms_per_molecule, molecule_count)
    system = create_scripted_torch_force(system, wrapped_model, cached_torchscript)

    integrator = mm.VerletIntegrator(TIMESTEP_FS * unit.femtosecond)
    platform = mm.Platform.getPlatformByName("CPU")

    t0 = time.perf_counter()
    context = mm.Context(system, integrator, platform)
    positions_nm = (all_positions_angstrom / 10.0) * unit.nanometer
    context.setPositions(positions_nm)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin)
    state = context.getState(getEnergy=True, getForces=True)
    initial_energy_kj_mol = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    compile_and_init_seconds = time.perf_counter() - t0

    integrator.step(WARMUP_STEPS)

    block_times = []
    for _ in range(MEASURE_BLOCKS):
        t1 = time.perf_counter()
        integrator.step(STEPS_PER_BLOCK)
        state = context.getState(getEnergy=True)
        _ = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        block_times.append(time.perf_counter() - t1)

    mean_block_seconds = statistics.mean(block_times)
    mean_step_seconds = mean_block_seconds / STEPS_PER_BLOCK
    std_step_seconds = statistics.pstdev(block_times) / STEPS_PER_BLOCK if len(block_times) > 1 else 0.0
    throughput = ns_per_day(mean_step_seconds, TIMESTEP_FS)

    report = {
        "env_python": sys.executable,
        "platform": platform.getName(),
        "input_pdb": str(INPUTS / "DME.pdb"),
        "generated_system_pdb": str(large_pdb),
        "cached_torchscript": str(cached_torchscript),
        "params": str(INPUTS / "params_sgnn.pickle"),
        "param_source": param_source,
        "target_atoms": TARGET_ATOMS,
        "atoms_per_molecule": atoms_per_molecule,
        "molecule_count": molecule_count,
        "box_length_angstrom": box_length,
        "compile_and_init_seconds": compile_and_init_seconds,
        "initial_energy_kj_per_mol": initial_energy_kj_mol,
        "warmup_steps": WARMUP_STEPS,
        "measure_blocks": MEASURE_BLOCKS,
        "steps_per_block": STEPS_PER_BLOCK,
        "timestep_fs": TIMESTEP_FS,
        "mean_step_seconds": mean_step_seconds,
        "std_step_seconds": std_step_seconds,
        "estimated_ns_per_day": throughput,
        "path_mode": "openmm_native_batched_monomer_torchforce",
    }

    (OUTPUTS / "benchmark_openmm_native_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (OUTPUTS / "benchmark_openmm_native_report.md").write_text(
        "\n".join(
            [
                "# OpenMM Native DME 10k Benchmark",
                "",
                f"- Python: {sys.executable}",
                f"- Platform: {platform.getName()}",
                f"- Molecules: {molecule_count}",
                f"- Atoms: {TARGET_ATOMS}",
                f"- Parameter source: {param_source}",
                f"- Timestep: {TIMESTEP_FS:.1f} fs",
                f"- Mean OpenMM step time: {mean_step_seconds:.6f} s",
                f"- Step time std: {std_step_seconds:.6f} s",
                f"- Init + compile time: {compile_and_init_seconds:.6f} s",
                f"- Estimated throughput: {throughput:.3f} ns/day",
                "- Path: native OpenMM `VerletIntegrator.step()` with `openmmtorch.TorchForce`.",
                "- Model: one DME sGNN model applied independently to each 16-atom DME block.",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
