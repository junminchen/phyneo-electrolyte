#!/usr/bin/env python3
"""Run OpenMM MD with MPIDForce from XML plus sGNNForceFast via openmmtorch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import openmm as mm
from openmm import app, unit
import openmmtorch
import torch
import torch.nn as nn

from openmm_mpid_sgnn_fast import (
    build_fast_bundle,
    describe_force,
    resolve_path,
    set_force_groups,
    topology_box_to_angstrom,
)

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))

from phyneo.utils import (
    find_residue_blocks,
    group_residue_blocks_by_name,
    non_residue_atom_indices,
    resolve_default_sgnn_specs,
    spec_for_residue_name,
)


class IndexedModelGroup(nn.Module):
    def __init__(self, model: nn.Module, index_groups: list[torch.Tensor]):
        super().__init__()
        self.model = model
        self.index_names: list[str] = []
        for i, indices in enumerate(index_groups):
            name = f"indices_{i}"
            self.register_buffer(name, indices.to(dtype=torch.long))
            self.index_names.append(name)

    def forward(self, positions_angstrom: torch.Tensor, box_angstrom: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float32, device=positions_angstrom.device)
        for name in self.index_names:
            indices = getattr(self, name)
            total = total + self.model(positions_angstrom.index_select(0, indices), box_angstrom)
        return total


class CombinedSGNNTorchForce(nn.Module):
    def __init__(
        self,
        standard_group: IndexedModelGroup | None,
        abn_groups: list[IndexedModelGroup],
    ):
        super().__init__()
        self.has_standard = standard_group is not None
        if standard_group is not None:
            self.standard_group = standard_group
        self.abn_groups = nn.ModuleList(abn_groups)

    def forward(self, positions_nm: torch.Tensor, boxvectors_nm: torch.Tensor) -> torch.Tensor:
        positions_angstrom = positions_nm.float() * 10.0
        box_angstrom = boxvectors_nm.float() * 10.0
        total = torch.zeros((), dtype=torch.float32, device=positions_angstrom.device)
        if self.has_standard:
            total = total + self.standard_group(positions_angstrom, box_angstrom)
        for group in self.abn_groups:
            total = total + group(positions_angstrom, box_angstrom)
        return total * 4.184


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenMM MD with MPIDForce + trained sGNNForceFast via openmmtorch"
    )
    parser.add_argument("--pdb", default="init.pdb", help="System PDB path")
    parser.add_argument("--xml", default="phyneo_ecl.xml", help="OpenMM XML with ADMPPmeForce/MPIDForce")
    parser.add_argument(
        "--standard-pdb",
        default="init_remaining.pdb",
        help="PDB used to build the non-ABn sGNN graph when ABn residues are present",
    )
    parser.add_argument("--pdb-bank", default="pdb_bank", help="Directory containing ABn template PDB files")
    parser.add_argument("--standard-params", default="params_sgnn.pickle", help="Checkpoint for the standard sGNN model")
    parser.add_argument("--abn-params", default="params_sgnn_ABn.pickle", help="Checkpoint for the ABn sGNN model")
    parser.add_argument("--steps", type=int, default=1000, help="Production MD steps")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup MD steps before timing")
    parser.add_argument("--dt-fs", type=float, default=1.0, help="Integrator timestep in fs")
    parser.add_argument("--temperature-k", type=float, default=298.0, help="Temperature in K")
    parser.add_argument("--friction-ps", type=float, default=1.0, help="Friction in ps^-1")
    parser.add_argument("--cutoff-nm", type=float, default=0.6, help="PME cutoff in nm")
    parser.add_argument("--platform", default="CUDA", help="OpenMM platform name")
    parser.add_argument("--device", default=None, help="Torch device used for tracing")
    parser.add_argument("--cuda-precision", default="mixed", help="CUDA precision mode")
    parser.add_argument("--report-interval", type=int, default=100, help="StateDataReporter interval")
    parser.add_argument("--output-pdb", default=None, help="Optional trajectory PDB reporter output")
    return parser.parse_args()


def build_combined_model(
    pdb: app.PDBFile,
    base_dir: Path,
    system_pdb_path: Path,
    standard_pdb_path: Path,
    pdb_bank_dir: Path,
    standard_params_path: Path,
    abn_params_path: Path,
    box_angstrom: torch.Tensor,
    device: torch.device,
):
    specs = resolve_default_sgnn_specs(base_dir)
    n_atoms = sum(1 for _ in pdb.topology.atoms())
    residue_blocks = find_residue_blocks(pdb.topology, specs["abn"].residue_names)
    residue_blocks_by_name = group_residue_blocks_by_name(residue_blocks)
    standard_atom_indices_np = non_residue_atom_indices(n_atoms, residue_blocks)

    standard_group = None
    if len(standard_atom_indices_np) > 0:
        standard_source = standard_pdb_path if residue_blocks else system_pdb_path
        standard_spec = specs["standard"]
        standard_bundle = build_fast_bundle(
            name="standard",
            pdb_path=standard_source,
            params_path=standard_params_path,
            nn_hops=standard_spec.nn,
            max_valence=standard_spec.max_valence,
            box_angstrom=box_angstrom,
            device=device,
            sigma=standard_spec.sigma,
            mu=standard_spec.mu,
            n_layers=standard_spec.n_layers,
            sizes=standard_spec.sizes,
        )
        if standard_bundle.atom_count != len(standard_atom_indices_np):
            raise ValueError("Standard sGNN graph atom count does not match selected atom count.")
        standard_indices = torch.tensor(standard_atom_indices_np, dtype=torch.long, device=device)
        standard_group = IndexedModelGroup(standard_bundle.model, [standard_indices])

    abn_groups: list[IndexedModelGroup] = []
    for residue_name, blocks in residue_blocks_by_name.items():
        spec = spec_for_residue_name(residue_name, specs)
        template_pdb = pdb_bank_dir / f"{residue_name}.pdb"
        bundle = build_fast_bundle(
            name=residue_name,
            pdb_path=template_pdb,
            params_path=abn_params_path,
            nn_hops=spec.nn,
            max_valence=spec.max_valence,
            box_angstrom=box_angstrom,
            device=device,
            sigma=spec.sigma,
            mu=spec.mu,
            n_layers=spec.n_layers,
            sizes=spec.sizes,
        )
        if bundle.atom_count != blocks[0].atom_count:
            raise ValueError(
                f"Template {template_pdb} has {bundle.atom_count} atoms but residue {residue_name} has {blocks[0].atom_count}."
            )
        abn_indices = [
            torch.arange(block.atom_start, block.atom_stop, dtype=torch.long, device=device)
            for block in blocks
        ]
        abn_groups.append(IndexedModelGroup(bundle.model, abn_indices))

    return CombinedSGNNTorchForce(standard_group, abn_groups), residue_blocks_by_name


def save_traced_model(
    model: nn.Module,
    positions_nm: torch.Tensor,
    box_nm: torch.Tensor,
) -> str:
    traced = torch.jit.trace(model, (positions_nm, box_nm))
    fd, temp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    traced.save(temp_path)
    return temp_path


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    pdb_path = resolve_path(base_dir, args.pdb)
    xml_path = resolve_path(base_dir, args.xml)
    standard_pdb_path = resolve_path(base_dir, args.standard_pdb)
    pdb_bank_dir = resolve_path(base_dir, args.pdb_bank)
    standard_params_path = resolve_path(base_dir, args.standard_params)
    abn_params_path = resolve_path(base_dir, args.abn_params)
    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print(f"OpenMM platform: {args.platform}")
    print(f"Torch trace device: {device}")

    pdb = app.PDBFile(str(pdb_path))
    box_angstrom = topology_box_to_angstrom(pdb.topology)
    model, residue_blocks_by_name = build_combined_model(
        pdb=pdb,
        base_dir=base_dir,
        system_pdb_path=pdb_path,
        standard_pdb_path=standard_pdb_path,
        pdb_bank_dir=pdb_bank_dir,
        standard_params_path=standard_params_path,
        abn_params_path=abn_params_path,
        box_angstrom=box_angstrom,
        device=device,
    )
    model = model.to(device).eval()

    if residue_blocks_by_name:
        found_names = ", ".join(
            f"{name} x{len(blocks)}" for name, blocks in sorted(residue_blocks_by_name.items())
        )
        print(f"ABn residues detected: {found_names}")
    else:
        print("ABn residues detected: none")

    positions_nm = torch.tensor(
        [[vec.x, vec.y, vec.z] for vec in pdb.positions],
        dtype=torch.float32,
        device=device,
    )
    box_nm = torch.tensor(
        [[vec.x, vec.y, vec.z] for vec in pdb.topology.getPeriodicBoxVectors()],
        dtype=torch.float32,
        device=device,
    )

    temp_path = save_traced_model(model, positions_nm, box_nm)
    try:
        ff = app.ForceField(str(xml_path))
        system = ff.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=args.cutoff_nm * unit.nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=True,
        )
        set_force_groups(system)

        torch_force = openmmtorch.TorchForce(temp_path)
        torch_force.setUsesPeriodicBoundaryConditions(True)
        torch_force.setForceGroup(1)
        system.addForce(torch_force)

        print("Forces in system:")
        for i in range(system.getNumForces()):
            print(f"  {i}: {describe_force(system.getForce(i))}")

        integrator = mm.LangevinIntegrator(
            args.temperature_k * unit.kelvin,
            args.friction_ps / unit.picosecond,
            args.dt_fs * unit.femtosecond,
        )
        platform = mm.Platform.getPlatformByName(args.platform)
        properties = {}
        if args.platform.upper() == "CUDA":
            properties["CudaPrecision"] = args.cuda_precision
        simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
        simulation.context.setPositions(pdb.positions)

        state_total = simulation.context.getState(getEnergy=True)
        state_mpid = simulation.context.getState(getEnergy=True, groups={0})
        state_sgnn = simulation.context.getState(getEnergy=True, groups={1})
        print(f"Initial total energy: {state_total.getPotentialEnergy()}")
        print(f"Initial MPID energy:  {state_mpid.getPotentialEnergy()}")
        print(f"Initial sGNN energy:  {state_sgnn.getPotentialEnergy()}")

        if args.report_interval > 0:
            simulation.reporters.append(
                app.StateDataReporter(
                    sys.stdout,
                    args.report_interval,
                    step=True,
                    potentialEnergy=True,
                    temperature=True,
                    speed=True,
                )
            )
        if args.output_pdb:
            simulation.reporters.append(app.PDBReporter(args.output_pdb, args.report_interval))

        if args.warmup > 0:
            print(f"Warmup: {args.warmup} steps")
            simulation.step(args.warmup)

        print(f"Production: {args.steps} steps")
        if args.steps > 0:
            simulation.step(args.steps)

        final_total = simulation.context.getState(getEnergy=True)
        final_mpid = simulation.context.getState(getEnergy=True, groups={0})
        final_sgnn = simulation.context.getState(getEnergy=True, groups={1})
        print(f"Final total energy: {final_total.getPotentialEnergy()}")
        print(f"Final MPID energy:  {final_mpid.getPotentialEnergy()}")
        print(f"Final sGNN energy:  {final_sgnn.getPotentialEnergy()}")
        return 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    raise SystemExit(main())
