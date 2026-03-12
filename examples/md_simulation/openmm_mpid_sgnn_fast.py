#!/usr/bin/env python3
"""Run OpenMM MD with MPIDForce from XML plus sGNNForceFast via CallbackPyForce.

This mirrors the residue splitting in client_dmff.py, but executes the long-range
electrostatics directly inside OpenMM and adds the trained sGNN correction as a
second force.

Current limitation: CallbackPyForce only passes positions, so the sGNN term uses
the initial periodic box from the PDB/XML setup. This is appropriate for fixed-cell
simulations and not for barostat-driven variable-cell runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import openmm as mm
from openmm import app, unit
import torch
from CallbackPyForce import Callable, TorchForce as CallbackTorchForce
import mpidplugin

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "examples" / "torch_gnn"))

from dmff.sgnn.graph import FSCALE_ANGLE, FSCALE_BOND, from_pdb as from_pdb_jax
from sgnn_fast import sGNNForceFast, load_params_from_pickle
from phyneo.utils import (
    find_residue_blocks,
    group_residue_blocks_by_name,
    non_residue_atom_indices,
    resolve_default_sgnn_specs,
    spec_for_residue_name,
)


@dataclass(frozen=True)
class FastModelBundle:
    name: str
    model: sGNNForceFast
    atom_count: int


class GraphAdapter:
    def __init__(self, graph):
        self.bonds = graph.bonds
        self.b0 = graph.b0
        self.fscale_bond = FSCALE_BOND
        self.angles = graph.angles
        self.cos_a0 = graph.cos_a0
        self.fscale_angle = FSCALE_ANGLE
        self.diheds = graph.diheds
        self.feature_atypes = graph.feature_atypes
        self.feature_indices = graph.feature_indices
        self.nb_connect = graph.nb_connect
        self.weights = graph.weights
        self.n_features = int(graph.n_features)
        self.max_valence = int(graph.max_valence)


class CombinedSGNNCallback:
    """Evaluate standard + ABn sGNN models on the current OpenMM coordinates."""

    def __init__(
        self,
        box_angstrom: torch.Tensor,
        device: torch.device,
        standard_bundle: FastModelBundle | None,
        standard_indices: torch.Tensor | None,
        abn_bundles: dict[str, FastModelBundle],
        abn_indices: dict[str, list[torch.Tensor]],
    ):
        self.box_angstrom = box_angstrom.to(device=device, dtype=torch.float32)
        self.device = device
        self.standard_bundle = standard_bundle
        self.standard_indices = standard_indices
        self.abn_bundles = abn_bundles
        self.abn_indices = abn_indices

    def __call__(self, positions_nm):
        positions = positions_nm.to(device=self.device, dtype=torch.float32) * 10.0
        total_energy = torch.zeros((), dtype=torch.float32, device=self.device)

        if self.standard_bundle is not None and self.standard_indices is not None:
            total_energy = total_energy + self.standard_bundle.model(
                positions.index_select(0, self.standard_indices),
                self.box_angstrom,
            )

        for residue_name, bundle in self.abn_bundles.items():
            for atom_indices in self.abn_indices[residue_name]:
                total_energy = total_energy + bundle.model(
                    positions.index_select(0, atom_indices),
                    self.box_angstrom,
                )

        return total_energy * 4.184


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenMM MD with MPIDForce + trained sGNNForceFast"
    )
    parser.add_argument("--pdb", default="init.pdb", help="System PDB path")
    parser.add_argument("--xml", default="phyneo_ecl.xml", help="OpenMM XML with ADMPPmeForce/MPIDForce")
    parser.add_argument(
        "--standard-pdb",
        default="init_remaining.pdb",
        help="PDB used to build the non-ABn sGNN graph when ABn residues are present",
    )
    parser.add_argument(
        "--pdb-bank",
        default="pdb_bank",
        help="Directory containing ABn template PDB files such as PF6.pdb",
    )
    parser.add_argument(
        "--standard-params",
        default="params_sgnn.pickle",
        help="Checkpoint for the standard sGNN model",
    )
    parser.add_argument(
        "--abn-params",
        default="params_sgnn_ABn.pickle",
        help="Checkpoint for the ABn sGNN model",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Production MD steps")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup MD steps before timing")
    parser.add_argument("--dt-fs", type=float, default=1.0, help="Integrator timestep in fs")
    parser.add_argument("--temperature-k", type=float, default=298.0, help="Temperature in K")
    parser.add_argument("--friction-ps", type=float, default=1.0, help="Friction in ps^-1")
    parser.add_argument("--cutoff-nm", type=float, default=0.6, help="PME cutoff in nm")
    parser.add_argument("--platform", default="CUDA", help="OpenMM platform name")
    parser.add_argument("--device", default=None, help="Torch device, defaults to cuda:0 if available")
    parser.add_argument("--cuda-precision", default="mixed", help="CUDA precision mode")
    parser.add_argument("--report-interval", type=int, default=100, help="StateDataReporter interval")
    parser.add_argument("--output-pdb", default=None, help="Optional trajectory PDB reporter output")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure MD throughput for the production steps",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_dir / path


def topology_box_to_angstrom(topology) -> torch.Tensor:
    box_vectors = topology.getPeriodicBoxVectors()
    if box_vectors is None:
        raise ValueError("Topology does not define periodic box vectors.")
    rows = [
        [vec.x, vec.y, vec.z]
        for vec in box_vectors
    ]
    return torch.tensor(rows, dtype=torch.float32) * 10.0


def prepare_graph(
    pdb_path: Path,
    nn_hops: int,
    max_valence: int,
    box_angstrom: torch.Tensor,
):
    graph = from_pdb_jax(str(pdb_path))
    if graph.box is None:
        graph.set_box(box_angstrom.detach().cpu().numpy())
    graph.get_all_subgraphs(nn_hops, typify=True)
    graph.prepare_subgraph_feature_calc(max_valence=max_valence)
    return graph


def build_fast_bundle(
    name: str,
    pdb_path: Path,
    params_path: Path,
    nn_hops: int,
    max_valence: int,
    box_angstrom: torch.Tensor,
    device: torch.device,
    sigma: float,
    mu: float,
    n_layers: tuple[int, int],
    sizes: tuple[tuple[int, ...], tuple[int, ...]],
) -> FastModelBundle:
    graph = prepare_graph(
        pdb_path,
        nn_hops=nn_hops,
        max_valence=max_valence,
        box_angstrom=box_angstrom,
    )
    model = sGNNForceFast(
        GraphAdapter(graph),
        n_layers=n_layers,
        sizes=[tuple(layer_sizes) for layer_sizes in sizes],
        nn_hops=nn_hops,
        sigma=sigma,
        mu=mu,
    )
    load_params_from_pickle(model, str(params_path))
    model = model.to(device)
    model.eval()
    return FastModelBundle(name=name, model=model, atom_count=graph.n_atoms)


def describe_force(force) -> str:
    if mpidplugin.MPIDForce.isinstance(force):
        return "MPIDForce"
    return force.__class__.__name__


def set_force_groups(system: mm.System) -> None:
    for force in system.getForces():
        if mpidplugin.MPIDForce.isinstance(force):
            force.setForceGroup(0)


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    pdb_path = resolve_path(base_dir, args.pdb)
    xml_path = resolve_path(base_dir, args.xml)
    standard_pdb_path = resolve_path(base_dir, args.standard_pdb)
    pdb_bank_dir = resolve_path(base_dir, args.pdb_bank)
    standard_params = resolve_path(base_dir, args.standard_params)
    abn_params = resolve_path(base_dir, args.abn_params)

    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"OpenMM platform: {args.platform}")
    print(f"Torch device: {device}")

    pdb = app.PDBFile(str(pdb_path))
    box_angstrom = topology_box_to_angstrom(pdb.topology)
    n_atoms = sum(1 for _ in pdb.topology.atoms())

    specs = resolve_default_sgnn_specs(base_dir)
    residue_blocks = find_residue_blocks(pdb.topology, specs["abn"].residue_names)
    residue_blocks_by_name = group_residue_blocks_by_name(residue_blocks)
    standard_atom_indices_np = non_residue_atom_indices(n_atoms, residue_blocks)
    standard_atom_indices = torch.tensor(standard_atom_indices_np, dtype=torch.long, device=device)

    standard_bundle = None
    if len(standard_atom_indices_np) > 0:
        standard_source = standard_pdb_path if residue_blocks else pdb_path
        standard_spec = specs["standard"]
        standard_bundle = build_fast_bundle(
            name="standard",
            pdb_path=standard_source,
            params_path=standard_params,
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
            raise ValueError(
                "Standard sGNN graph atom count does not match the non-ABn atom selection."
            )

    abn_bundles: dict[str, FastModelBundle] = {}
    abn_indices: dict[str, list[torch.Tensor]] = {}
    for residue_name, blocks in residue_blocks_by_name.items():
        spec = spec_for_residue_name(residue_name, specs)
        template_pdb = pdb_bank_dir / f"{residue_name}.pdb"
        bundle = build_fast_bundle(
            name=residue_name,
            pdb_path=template_pdb,
            params_path=abn_params,
            nn_hops=spec.nn,
            max_valence=spec.max_valence,
            box_angstrom=box_angstrom,
            device=device,
            sigma=spec.sigma,
            mu=spec.mu,
            n_layers=spec.n_layers,
            sizes=spec.sizes,
        )
        atom_count = blocks[0].atom_count
        if bundle.atom_count != atom_count:
            raise ValueError(
                f"Template {template_pdb} has {bundle.atom_count} atoms but residue {residue_name} has {atom_count}."
            )
        abn_bundles[residue_name] = bundle
        abn_indices[residue_name] = [
            torch.arange(block.atom_start, block.atom_stop, dtype=torch.long, device=device)
            for block in blocks
        ]

    if residue_blocks:
        found_names = ", ".join(
            f"{name} x{len(blocks)}" for name, blocks in sorted(residue_blocks_by_name.items())
        )
        print(f"ABn residues detected: {found_names}")
    else:
        print("ABn residues detected: none")

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

    callback_model = CombinedSGNNCallback(
        box_angstrom=box_angstrom,
        device=device,
        standard_bundle=standard_bundle,
        standard_indices=standard_atom_indices if standard_bundle is not None else None,
        abn_bundles=abn_bundles,
        abn_indices=abn_indices,
    )
    callback = Callable(id(callback_model), Callable.RETURN_ENERGY)
    sgnn_force = CallbackTorchForce(callback)
    sgnn_force.setForceGroup(1)
    system.addForce(sgnn_force)

    print("Forces in system:")
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        print(f"  {i}: {describe_force(force)}")

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
        t0 = time.perf_counter()
        simulation.step(args.steps)
        dt = time.perf_counter() - t0
        if args.benchmark:
            ms_step = dt / args.steps * 1000.0
            ns_day = args.steps * args.dt_fs * 1e-6 / dt * 86400.0
            print(f"Benchmark time:    {dt:.3f} s")
            print(f"Benchmark ms/step: {ms_step:.3f}")
            print(f"Benchmark ns/day:  {ns_day:.3f}")

    final_total = simulation.context.getState(getEnergy=True)
    final_mpid = simulation.context.getState(getEnergy=True, groups={0})
    final_sgnn = simulation.context.getState(getEnergy=True, groups={1})
    print(f"Final total energy: {final_total.getPotentialEnergy()}")
    print(f"Final MPID energy:  {final_mpid.getPotentialEnergy()}")
    print(f"Final sGNN energy:  {final_sgnn.getPotentialEnergy()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
