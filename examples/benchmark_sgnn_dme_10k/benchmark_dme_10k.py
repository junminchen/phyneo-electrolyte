#!/usr/bin/env python

from __future__ import annotations

import json
import math
import pickle
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad, vmap
from openmm.app import PDBFile

from dmff.sgnn.gnn import MolGNNForce
from dmff.sgnn.graph import from_pdb


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
TARGET_ATOMS = 10_000
REPEATS = 20
WARMUP_RUNS = 3
GRID_SPACING_ANGSTROM = 9.0
TIMESTEPS_FS = (0.5, 1.0, 2.0)


def load_positions_from_pdb(pdb_path: Path) -> tuple[np.ndarray, list[tuple[str, str]]]:
    pdb = PDBFile(str(pdb_path))
    positions_angstrom = np.asarray(pdb.positions._value, dtype=np.float64) * 10.0
    atom_records = []
    for atom in pdb.topology.atoms():
        atom_records.append((atom.name, atom.element.symbol if atom.element else "X"))
    return positions_angstrom, atom_records


def build_replicated_system(
    monomer_positions: np.ndarray,
    target_atoms: int,
    spacing_angstrom: float,
) -> tuple[np.ndarray, np.ndarray]:
    atoms_per_molecule = monomer_positions.shape[0]
    if target_atoms % atoms_per_molecule != 0:
        raise ValueError(
            f"target_atoms={target_atoms} is not divisible by atoms_per_molecule={atoms_per_molecule}"
        )

    molecule_count = target_atoms // atoms_per_molecule
    grid_side = math.ceil(molecule_count ** (1.0 / 3.0))
    offsets = []
    for ix in range(grid_side):
        for iy in range(grid_side):
            for iz in range(grid_side):
                offsets.append(np.array([ix, iy, iz], dtype=np.float64) * spacing_angstrom)
                if len(offsets) == molecule_count:
                    break
            if len(offsets) == molecule_count:
                break
        if len(offsets) == molecule_count:
            break

    batched_positions = np.stack([monomer_positions + offset for offset in offsets], axis=0)
    box_length = grid_side * spacing_angstrom + spacing_angstrom
    box = np.eye(3, dtype=np.float64) * box_length
    return batched_positions, box


def write_pdb(
    output_path: Path,
    batched_positions: np.ndarray,
    atom_records: list[tuple[str, str]],
    box: np.ndarray,
) -> None:
    residue_name = "DME"
    with output_path.open("w", encoding="ascii") as handle:
        handle.write("HEADER    DME 10K SGNN BENCHMARK\n")
        handle.write(
            "CRYST1"
            f"{box[0,0]:9.3f}{box[1,1]:9.3f}{box[2,2]:9.3f}"
            "  90.00  90.00  90.00 P 1           1\n"
        )
        serial = 1
        for residue_index, molecule_positions in enumerate(batched_positions, start=1):
            residue_id = residue_index % 10000
            for (atom_name, element), position in zip(atom_records, molecule_positions):
                handle.write(
                    f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} A{residue_id:4d}"
                    f"    {position[0]:8.3f}{position[1]:8.3f}{position[2]:8.3f}"
                    f"  1.00  0.00          {element:>2s}\n"
                )
                serial += 1
        handle.write("END\n")


def load_params(params_path: Path):
    with params_path.open("rb") as handle:
        return pickle.load(handle)


def tree_shapes(tree):
    return jax.tree_util.tree_map(lambda value: tuple(value.shape), tree)


def select_params(model, params_path: Path):
    loaded_params = load_params(params_path)
    default_params = model.params
    if tree_shapes(loaded_params) == tree_shapes(default_params):
        return loaded_params, "loaded"
    return default_params, "model_init"


def ns_per_day(step_time_seconds: float, timestep_fs: float) -> float:
    return 86_400.0 * timestep_fs / (step_time_seconds * 1.0e6)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    monomer_pdb = INPUTS / "DME.pdb"
    params_path = INPUTS / "params_sgnn.pickle"

    monomer_positions, atom_records = load_positions_from_pdb(monomer_pdb)
    batched_positions, box = build_replicated_system(
        monomer_positions,
        target_atoms=TARGET_ATOMS,
        spacing_angstrom=GRID_SPACING_ANGSTROM,
    )
    system_pdb = OUTPUTS / "dme_10000_atoms.pdb"
    write_pdb(system_pdb, batched_positions, atom_records, box)

    graph = from_pdb(str(monomer_pdb))
    model = MolGNNForce(graph, nn=1)
    params, param_source = select_params(model, params_path)

    batched_positions_jax = jnp.array(batched_positions)
    box_jax = jnp.array(box)
    batch_forward = vmap(model.forward, in_axes=(0, None, None), out_axes=0)

    def total_energy(positions: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(batch_forward(positions, box_jax, params))

    compiled_energy_force = jit(value_and_grad(total_energy))

    t0 = time.perf_counter()
    warmup_energy, warmup_grad = compiled_energy_force(batched_positions_jax)
    warmup_energy.block_until_ready()
    warmup_grad.block_until_ready()
    compile_time_seconds = time.perf_counter() - t0

    warmup_times = []
    for _ in range(WARMUP_RUNS):
        t1 = time.perf_counter()
        energy, grad = compiled_energy_force(batched_positions_jax)
        energy.block_until_ready()
        grad.block_until_ready()
        warmup_times.append(time.perf_counter() - t1)

    run_times = []
    energies = []
    for _ in range(REPEATS):
        t1 = time.perf_counter()
        energy, grad = compiled_energy_force(batched_positions_jax)
        energy.block_until_ready()
        grad.block_until_ready()
        dt = time.perf_counter() - t1
        run_times.append(dt)
        energies.append(float(energy))

    mean_step_seconds = statistics.mean(run_times)
    std_step_seconds = statistics.pstdev(run_times) if len(run_times) > 1 else 0.0
    mean_warmup_seconds = statistics.mean(warmup_times)
    atoms_per_molecule = monomer_positions.shape[0]
    molecule_count = batched_positions.shape[0]
    ns_day_map = {
        f"{timestep_fs:.1f}_fs": ns_per_day(mean_step_seconds, timestep_fs)
        for timestep_fs in TIMESTEPS_FS
    }

    report = {
        "env_python": sys.executable,
        "jax_devices": [str(device) for device in jax.devices()],
        "monomer_pdb": str(monomer_pdb),
        "params": str(params_path),
        "param_source": param_source,
        "generated_system_pdb": str(system_pdb),
        "target_atoms": TARGET_ATOMS,
        "atoms_per_molecule": atoms_per_molecule,
        "molecule_count": molecule_count,
        "box_length_angstrom": float(box[0, 0]),
        "compile_time_seconds": compile_time_seconds,
        "mean_warmup_seconds": mean_warmup_seconds,
        "mean_step_seconds": mean_step_seconds,
        "std_step_seconds": std_step_seconds,
        "repeats": REPEATS,
        "energy_kcal_per_mol_like": statistics.mean(energies),
        "ns_per_day": ns_day_map,
    }

    report_json = OUTPUTS / "benchmark_report.json"
    report_md = OUTPUTS / "benchmark_report.md"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md.write_text(
        "\n".join(
            [
                "# DME 10k-Atom sGNN Benchmark",
                "",
                f"- Device(s): {', '.join(report['jax_devices'])}",
                f"- Python: {sys.executable}",
                f"- Molecules: {molecule_count}",
                f"- Atoms: {TARGET_ATOMS}",
                f"- Parameter source: {param_source}",
                f"- Mean energy+force step time: {mean_step_seconds:.6f} s",
                f"- Step time std: {std_step_seconds:.6f} s",
                f"- JIT compile time: {compile_time_seconds:.6f} s",
                f"- Estimated throughput at 0.5 fs: {ns_day_map['0.5_fs']:.3f} ns/day",
                f"- Estimated throughput at 1.0 fs: {ns_day_map['1.0_fs']:.3f} ns/day",
                f"- Estimated throughput at 2.0 fs: {ns_day_map['2.0_fs']:.3f} ns/day",
                "- Note: timing uses pure sGNN energy+force evaluation only.",
                "- Note: full MD throughput will be lower once nonbonded and integrator costs are included.",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
