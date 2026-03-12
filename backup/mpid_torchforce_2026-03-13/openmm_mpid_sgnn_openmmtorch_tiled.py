#!/usr/bin/env python3
"""Run OpenMM MD with MPIDForce plus tiled fixed-topology sGNN via openmmtorch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time
import warnings

import numpy as np


def _ensure_torch_lib_path() -> None:
    try:
        import torch as _torch
    except ImportError:
        return

    candidate_dirs = [
        Path(_torch.__file__).resolve().parent / "lib",
        Path(_torch.__file__).resolve().parent.parent / "nvidia" / "cufile" / "lib",
    ]
    existing = [str(path) for path in candidate_dirs if path.exists()]
    if not existing:
        return

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [part for part in ld_path.split(os.pathsep) if part]
    missing = [path for path in existing if path not in current_parts]
    if not missing:
        return

    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([*missing, *current_parts])
    warnings.warn(
        "Added Torch/OpenMM runtime libraries to LD_LIBRARY_PATH for openmmtorch.",
        stacklevel=2,
    )


_ensure_torch_lib_path()

import openmm as mm
from openmm import app, unit
import openmmtorch
import torch
import torch.nn as nn

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "examples" / "torch_gnn"))

from dmff.sgnn.graph import ATYPE_INDEX, FSCALE_ANGLE, FSCALE_BOND, TopGraph
from sgnn_fast import load_params_from_pickle
from openmm_mpid_sgnn_fast import describe_force, resolve_path, set_force_groups, topology_box_to_angstrom
from phyneo.utils import resolve_default_sgnn_specs


class TiledSGNNGroup(nn.Module):
    def __init__(
        self,
        template_graph,
        residue_starts: list[int],
        sigma: float,
        mu: float,
        mu_multiplier: int,
        n_layers: tuple[int, int],
        sizes: tuple[tuple[int, ...], tuple[int, ...]],
    ):
        super().__init__()
        self.sigma = float(sigma)
        self.mu = float(mu)
        self.mu_multiplier = float(mu_multiplier)
        self.max_valence = int(template_graph.max_valence)
        self.nn_hops = int(template_graph.nn)

        n_copies = len(residue_starts)
        self.num_bonds = int(len(template_graph.bonds))
        self.num_angles = int(len(template_graph.angles))
        self.num_diheds = int(len(template_graph.diheds))

        def to_long(arr):
            return torch.tensor(np.array(arr), dtype=torch.long)

        def to_float(arr):
            return torch.tensor(np.array(arr), dtype=torch.float32)

        def tile_ic_map(idx, n_mols: int, ic_count: int):
            idx_t = to_long(idx)
            tiled = []
            for i in range(n_mols):
                tiled.append(torch.where(idx_t >= 0, idx_t + i * ic_count, idx_t))
            return torch.cat(tiled, dim=0)

        def tile_atom_map(atom_indices, starts: list[int]):
            atom_t = to_long(atom_indices)
            tiled = []
            for start in starts:
                tiled.append(atom_t + int(start))
            if tiled:
                return torch.cat(tiled, dim=0)
            return torch.zeros((0, atom_t.shape[-1]), dtype=torch.long)

        self.register_buffer("bonds_atoms", tile_atom_map(template_graph.bonds, residue_starts))
        self.register_buffer("angles_atoms", tile_atom_map(template_graph.angles, residue_starts))
        self.register_buffer("diheds_atoms", tile_atom_map(template_graph.diheds, residue_starts))
        self.register_buffer("b0", torch.tile(to_float(template_graph.b0), (n_copies,)))
        self.register_buffer("cos_a0", torch.tile(to_float(template_graph.cos_a0), (n_copies,)))
        self.register_buffer("feature_atypes", torch.tile(to_float(template_graph.feature_atypes), (n_copies, 1, 1)))
        self.register_buffer("weights", torch.tile(to_float(template_graph.weights), (n_copies,)))

        if self.nn_hops == 1:
            self.register_buffer("nb_connect", torch.tile(to_float(template_graph.nb_connect), (n_copies, 1)))
        else:
            self.register_buffer("nb_connect", torch.zeros((0, 0), dtype=torch.float32))

        self.register_buffer("idx_bonds", tile_ic_map(template_graph.feature_indices["bonds"], n_copies, self.num_bonds))
        self.register_buffer("idx_angles0", tile_ic_map(template_graph.feature_indices["angles0"], n_copies, self.num_angles))
        self.register_buffer("idx_angles1", tile_ic_map(template_graph.feature_indices["angles1"], n_copies, self.num_angles))
        self.register_buffer("idx_diheds", tile_ic_map(template_graph.feature_indices["diheds"], n_copies, self.num_diheds))

        self.w = nn.Parameter(torch.randn(1))
        self.fc0 = nn.ModuleList()
        dim_in = int(template_graph.n_features)
        for i_layer in range(n_layers[0]):
            dim_out = int(sizes[0][i_layer])
            self.fc0.append(nn.Linear(dim_in, dim_out))
            dim_in = dim_out

        self.fc1 = nn.ModuleList()
        for i_layer in range(n_layers[1]):
            dim_out = int(sizes[1][i_layer])
            self.fc1.append(nn.Linear(dim_in, dim_out))
            dim_in = dim_out
        self.fc_final = nn.Linear(dim_in, 1)

    def _pbc_shift(self, dr: torch.Tensor, box: torch.Tensor, box_inv: torch.Tensor) -> torch.Tensor:
        shift = torch.round(torch.matmul(dr, box_inv))
        return dr - torch.matmul(shift, box)

    def forward(self, positions_angstrom: torch.Tensor, box_angstrom: torch.Tensor) -> torch.Tensor:
        box_inv = torch.linalg.inv(box_angstrom)

        if self.num_bonds > 0:
            pos0 = positions_angstrom[self.bonds_atoms[:, 0]]
            pos1 = positions_angstrom[self.bonds_atoms[:, 1]]
            dr = self._pbc_shift(pos1 - pos0, box_angstrom, box_inv)
            fb = (torch.linalg.norm(dr, dim=1) - self.b0) * FSCALE_BOND
        else:
            fb = torch.zeros(0, dtype=torch.float32, device=positions_angstrom.device)

        if self.num_angles > 0:
            rj = positions_angstrom[self.angles_atoms[:, 0]]
            ri = positions_angstrom[self.angles_atoms[:, 1]]
            rk = positions_angstrom[self.angles_atoms[:, 2]]
            r_ij = self._pbc_shift(rj - ri, box_angstrom, box_inv)
            r_ik = self._pbc_shift(rk - ri, box_angstrom, box_inv)
            cos_a = torch.sum(r_ij * r_ik, dim=1) / (
                torch.linalg.norm(r_ij, dim=1) * torch.linalg.norm(r_ik, dim=1) + 1e-10
            )
            fa = (cos_a - self.cos_a0) * FSCALE_ANGLE
        else:
            fa = torch.zeros(0, dtype=torch.float32, device=positions_angstrom.device)

        if self.num_diheds > 0:
            ri = positions_angstrom[self.diheds_atoms[:, 0]]
            rj = positions_angstrom[self.diheds_atoms[:, 1]]
            rk = positions_angstrom[self.diheds_atoms[:, 2]]
            rl = positions_angstrom[self.diheds_atoms[:, 3]]
            r_jk = self._pbc_shift(rk - rj, box_angstrom, box_inv)
            r_ji = self._pbc_shift(ri - rj, box_angstrom, box_inv)
            r_kl = self._pbc_shift(rl - rk, box_angstrom, box_inv)
            n1 = torch.cross(r_jk, r_ji, dim=1)
            n2 = torch.cross(r_kl, -r_jk, dim=1)
            fd = torch.sum(n1 * n2, dim=1) / (
                torch.linalg.norm(n1, dim=1) * torch.linalg.norm(n2, dim=1) + 1e-10
            )
        else:
            fd = torch.zeros(0, dtype=torch.float32, device=positions_angstrom.device)

        f_bonds = torch.zeros_like(self.idx_bonds, dtype=torch.float32)
        mask_bonds = self.idx_bonds >= 0
        if self.num_bonds > 0:
            f_bonds[mask_bonds] = fb[self.idx_bonds[mask_bonds]]

        f_angles0 = torch.zeros_like(self.idx_angles0, dtype=torch.float32)
        mask_angles0 = self.idx_angles0 >= 0
        if self.num_angles > 0:
            f_angles0[mask_angles0] = fa[self.idx_angles0[mask_angles0]]

        f_angles1 = torch.zeros_like(self.idx_angles1, dtype=torch.float32)
        mask_angles1 = self.idx_angles1 >= 0
        if self.num_angles > 0:
            f_angles1[mask_angles1] = fa[self.idx_angles1[mask_angles1]]

        f_diheds = torch.zeros_like(self.idx_diheds, dtype=torch.float32)
        mask_diheds = self.idx_diheds >= 0
        if self.num_diheds > 0:
            f_diheds[mask_diheds] = fd[self.idx_diheds[mask_diheds]]

        features = torch.cat(
            [self.feature_atypes, f_bonds, f_angles0, f_angles1, f_diheds],
            dim=-1,
        )

        for layer in self.fc0:
            features = torch.tanh(layer(features))

        if self.nn_hops == 1:
            mv = self.max_valence
            nb_connect0 = self.nb_connect[:, : mv - 1]
            nb_connect1 = self.nb_connect[:, mv - 1 : 2 * (mv - 1)]
            nb0 = torch.sum(nb_connect0, dim=-1, keepdim=True)
            nb1 = torch.sum(nb_connect1, dim=-1, keepdim=True)

            f_center = features[:, 0, :]
            f_nb0 = features[:, 1:mv, :]
            f_nb1 = features[:, mv : 2 * mv - 1, :]

            sum_nb0 = torch.bmm(nb_connect0.unsqueeze(1), f_nb0).squeeze(1)
            sum_nb1 = torch.bmm(nb_connect1.unsqueeze(1), f_nb1).squeeze(1)

            h0 = (nb0 > 0.5).float()
            h1 = (nb1 > 0.5).float()
            nb0_safe = torch.where(nb0 < 1e-5, torch.ones_like(nb0) * 1e-5, nb0)
            nb1_safe = torch.where(nb1 < 1e-5, torch.ones_like(nb1) * 1e-5, nb1)

            features = (
                f_center * (1.0 - h0 * self.w - h1 * self.w)
                + (self.w * sum_nb0) / nb0_safe
                + (self.w * sum_nb1) / nb1_safe
            )
        else:
            features = features[:, 0, :]

        for layer in self.fc1:
            features = torch.tanh(layer(features))
        energies = self.fc_final(features).squeeze(-1)
        raw = torch.sum(self.weights * energies)
        return raw * self.sigma + self.mu * self.mu_multiplier


class CombinedTiledSGNNForce(nn.Module):
    def __init__(self, standard_mu: float, standard_groups: list[TiledSGNNGroup], abn_groups: list[TiledSGNNGroup]):
        super().__init__()
        self.standard_mu = float(standard_mu)
        self.standard_groups = nn.ModuleList(standard_groups)
        self.abn_groups = nn.ModuleList(abn_groups)

    def forward(self, positions_nm: torch.Tensor, boxvectors_nm: torch.Tensor) -> torch.Tensor:
        positions_angstrom = positions_nm.float() * 10.0
        box_angstrom = boxvectors_nm.float() * 10.0
        total = torch.zeros((), dtype=torch.float32, device=positions_angstrom.device)
        if len(self.standard_groups) > 0:
            total = total + positions_angstrom.sum() * 0.0 + self.standard_mu
        for group in self.standard_groups:
            total = total + group(positions_angstrom, box_angstrom)
        for group in self.abn_groups:
            total = total + group(positions_angstrom, box_angstrom)
        return total * 4.184


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenMM MD with MPIDForce + tiled fixed-topology sGNN via openmmtorch")
    parser.add_argument("--pdb", default="init.pdb", help="System PDB path")
    parser.add_argument("--xml", default="phyneo_ecl.xml", help="OpenMM XML with ADMPPmeForce/MPIDForce")
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
    parser.add_argument("--benchmark", action="store_true", help="Measure MD throughput")
    return parser.parse_args()


def residue_positions_angstrom(pdb: app.PDBFile, residue) -> np.ndarray:
    pos = np.asarray(pdb.positions._value, dtype=np.float32) * 10.0
    atom_indices = [atom.index for atom in residue.atoms()]
    return pos[atom_indices]


def build_template_graph(pdb: app.PDBFile, residue_name: str, spec, box_angstrom: torch.Tensor):
    residues = [res for res in pdb.topology.residues() if res.name == residue_name]
    if not residues:
        raise ValueError(f"No residue named {residue_name} found in topology.")
    residue = residues[0]
    atom_indices = [atom.index for atom in residue.atoms()]
    index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(atom_indices)}

    elements = np.array([atom.element.symbol for atom in residue.atoms()], dtype=object)
    bonds = []
    for bond in pdb.topology.bonds():
        i = bond[0].index
        j = bond[1].index
        if i in index_map and j in index_map:
            bonds.append(np.sort([index_map[i], index_map[j]]))
    if bonds:
        bonds = np.array(bonds, dtype=int)
    else:
        bonds = np.zeros((0, 2), dtype=int)

    graph = TopGraph(elements, bonds, positions=residue_positions_angstrom(pdb, residue), box=box_angstrom.detach().cpu().numpy())
    graph.get_all_subgraphs(spec.nn, typify=True)
    graph.prepare_subgraph_feature_calc(max_valence=spec.max_valence, atype_index=ATYPE_INDEX)
    return graph


def residue_starts_by_name(topology) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for residue in topology.residues():
        atom_indices = [atom.index for atom in residue.atoms()]
        if not atom_indices:
            continue
        grouped.setdefault(residue.name, []).append(atom_indices[0])
    return grouped


def build_model(pdb: app.PDBFile, box_angstrom: torch.Tensor, device: torch.device, standard_params_path: Path, abn_params_path: Path):
    specs = resolve_default_sgnn_specs(Path(__file__).resolve().parent)
    starts = residue_starts_by_name(pdb.topology)

    standard_groups: list[TiledSGNNGroup] = []
    standard_mu = specs["standard"].mu
    for residue_name in ("DMC", "ECA"):
        if residue_name not in starts:
            continue
        graph = build_template_graph(pdb, residue_name, specs["standard"], box_angstrom)
        group = TiledSGNNGroup(
            graph,
            residue_starts=starts[residue_name],
            sigma=specs["standard"].sigma,
            mu=0.0,
            mu_multiplier=0,
            n_layers=specs["standard"].n_layers,
            sizes=specs["standard"].sizes,
        )
        load_params_from_pickle(group, str(standard_params_path))
        standard_groups.append(group.to(device).eval())

    abn_groups: list[TiledSGNNGroup] = []
    for residue_name in specs["abn"].residue_names:
        if residue_name not in starts:
            continue
        graph = build_template_graph(pdb, residue_name, specs["abn"], box_angstrom)
        group = TiledSGNNGroup(
            graph,
            residue_starts=starts[residue_name],
            sigma=specs["abn"].sigma,
            mu=specs["abn"].mu,
            mu_multiplier=len(starts[residue_name]),
            n_layers=specs["abn"].n_layers,
            sizes=specs["abn"].sizes,
        )
        load_params_from_pickle(group, str(abn_params_path))
        abn_groups.append(group.to(device).eval())

    return CombinedTiledSGNNForce(standard_mu, standard_groups, abn_groups).to(device).eval()


def save_scripted_model(model: nn.Module, positions_nm: torch.Tensor, box_nm: torch.Tensor) -> str:
    scripted = torch.jit.trace(model, (positions_nm, box_nm))
    fd, temp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    scripted.save(temp_path)
    return temp_path


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    pdb_path = resolve_path(base_dir, args.pdb)
    xml_path = resolve_path(base_dir, args.xml)
    standard_params_path = resolve_path(base_dir, args.standard_params)
    abn_params_path = resolve_path(base_dir, args.abn_params)
    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print(f"OpenMM platform: {args.platform}")
    print(f"Torch trace device: {device}")

    pdb = app.PDBFile(str(pdb_path))
    box_angstrom = topology_box_to_angstrom(pdb.topology)
    model = build_model(pdb, box_angstrom, device, standard_params_path, abn_params_path)

    positions_nm = torch.tensor([[vec.x, vec.y, vec.z] for vec in pdb.positions], dtype=torch.float32, device=device)
    box_nm = torch.tensor([[vec.x, vec.y, vec.z] for vec in pdb.topology.getPeriodicBoxVectors()], dtype=torch.float32, device=device)

    temp_path = save_scripted_model(model, positions_nm, box_nm)
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

        if args.warmup > 0:
            print(f"Warmup: {args.warmup} steps")
            simulation.step(args.warmup)

        print(f"Production: {args.steps} steps")
        if args.steps > 0:
            t0 = time.perf_counter()
            simulation.step(args.steps)
            dt = time.perf_counter() - t0
            if args.benchmark:
                print(f"Benchmark time:    {dt:.3f} s")
                print(f"Benchmark ms/step: {dt / args.steps * 1000.0:.3f}")
                print(f"Benchmark ns/day:  {args.steps * args.dt_fs * 1e-6 / dt * 86400.0:.3f}")

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
