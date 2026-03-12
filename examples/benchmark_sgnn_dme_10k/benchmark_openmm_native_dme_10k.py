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
    atoms = [(a.name, a.element.symbol) for a in pdb.topology.atoms()]
    bonds = [(b[0].index, b[1].index) for b in pdb.topology.bonds()]
    return atoms, bonds, positions_angstrom


def build_replicated_positions():
    atoms, bonds, monomer_positions = load_monomer()
    atoms_per_molecule = len(atoms)
    if TARGET_ATOMS % atoms_per_molecule != 0:
        raise ValueError("TARGET_ATOMS must be divisible by atoms per DME molecule")
    molecule_count = TARGET_ATOMS // atoms_per_molecule

    grid_side = math.ceil(molecule_count ** (1.0 / 3.0))
    offsets = []
    for ix in range(grid_side):
        for iy in range(grid_side):
            for iz in range(grid_side):
                offsets.append(np.array([ix, iy, iz], dtype=np.float64) * GRID_SPACING_ANGSTROM)
    offsets = offsets[:molecule_count]

    all_positions = np.concatenate([monomer_positions + offset for offset in offsets], axis=0)
    box_length = grid_side * GRID_SPACING_ANGSTROM
    return all_positions, atoms_per_molecule, molecule_count, box_length, atoms, bonds


def load_sgnn_params(model: nn.Module, pickle_path: Path) -> str:
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
    
    model.load_state_dict(state_dict)
    return "loaded"


# Vectorized Custom Model for Benchmark
class sGNNForceOptimized(nn.Module):
    def __init__(self, n_layers, sizes, n_features, 
                 tiled_bonds_atoms, tiled_b0,
                 tiled_angles_atoms, tiled_cos_a0,
                 tiled_diheds_atoms,
                 tiled_idx_bonds, tiled_idx_angles0, tiled_idx_angles1, tiled_idx_diheds,
                 tiled_feature_atypes, tiled_weights, tiled_nb_connect):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1))
        self.fc0 = nn.ModuleList([nn.Linear(n_features if i==0 else sizes[0][i-1], sizes[0][i]) for i in range(n_layers[0])])
        self.fc1 = nn.ModuleList([nn.Linear(sizes[0][-1] if i==0 else sizes[1][i-1], sizes[1][i]) for i in range(n_layers[1])])
        self.fc_final = nn.Linear(sizes[1][-1], 1)
        self.sigma = 162.13039087945623
        self.mu = 117.41975505778706

        # Register all tiling buffers
        self.register_buffer("t_bonds_atoms", tiled_bonds_atoms)
        self.register_buffer("t_b0", tiled_b0)
        self.register_buffer("t_angles_atoms", tiled_angles_atoms)
        self.register_buffer("t_cos_a0", tiled_cos_a0)
        self.register_buffer("t_diheds_atoms", tiled_diheds_atoms)
        self.register_buffer("t_idx_bonds", tiled_idx_bonds)
        self.register_buffer("t_idx_angles0", tiled_idx_angles0)
        self.register_buffer("t_idx_angles1", tiled_idx_angles1)
        self.register_buffer("t_idx_diheds", tiled_idx_diheds)
        self.register_buffer("t_feature_atypes", tiled_feature_atypes)
        self.register_buffer("t_weights", tiled_weights)
        self.register_buffer("t_nb_connect", tiled_nb_connect)

    def forward(self, pos, boxvectors):
        # OpenMM units: nm to angstrom
        pos = pos.float() * 10.0
        box = boxvectors.float() * 10.0
        box_inv = torch.linalg.inv(box)
        
        # 1. Calc ICs
        # Bonds
        dr_b = pos[self.t_bonds_atoms[:,1]] - pos[self.t_bonds_atoms[:,0]]
        ds_b = torch.matmul(dr_b, box_inv.T)
        dr_b = torch.matmul(ds_b - torch.floor(ds_b + 0.5), box)
        bl = torch.norm(dr_b, dim=1)
        fb = (bl - self.t_b0) * 10.0
        
        # Angles
        rj, ri, rk = pos[self.t_angles_atoms[:,0]], pos[self.t_angles_atoms[:,1]], pos[self.t_angles_atoms[:,2]]
        dr_ij = torch.matmul(torch.matmul(rj-ri, box_inv.T) - torch.floor(torch.matmul(rj-ri, box_inv.T)+0.5), box)
        dr_ik = torch.matmul(torch.matmul(rk-ri, box_inv.T) - torch.floor(torch.matmul(rk-ri, box_inv.T)+0.5), box)
        cos_a = torch.sum(dr_ij*dr_ik, dim=1) / (torch.norm(dr_ij, dim=1)*torch.norm(dr_ik, dim=1) + 1e-8)
        fa = (cos_a - self.t_cos_a0) * 5.0
        
        # Diheds (Simplified)
        ri, rj, rk, rl = pos[self.t_diheds_atoms[:,0]], pos[self.t_diheds_atoms[:,1]], pos[self.t_diheds_atoms[:,2]], pos[self.t_diheds_atoms[:,3]]
        r_jk = torch.matmul(torch.matmul(rk-rj, box_inv.T) - torch.floor(torch.matmul(rk-rj, box_inv.T)+0.5), box)
        r_ji = torch.matmul(torch.matmul(ri-rj, box_inv.T) - torch.floor(torch.matmul(ri-rj, box_inv.T)+0.5), box)
        r_kl = torch.matmul(torch.matmul(rl-rk, box_inv.T) - torch.floor(torch.matmul(rl-rk, box_inv.T)+0.5), box)
        n1, n2 = torch.linalg.cross(r_jk, r_ji), torch.linalg.cross(r_kl, -r_jk)
        fd = torch.sum(n1*n2, dim=1) / (torch.norm(n1, dim=1)*torch.norm(n2, dim=1) + 1e-8)

        # 2. Features
        f_b = torch.zeros_like(self.t_idx_bonds, dtype=torch.float32)
        m_b = self.t_idx_bonds >= 0
        f_b[m_b] = fb[self.t_idx_bonds[m_b]]
        
        f_a0 = torch.zeros_like(self.t_idx_angles0, dtype=torch.float32)
        m_a0 = self.t_idx_angles0 >= 0
        f_a0[m_a0] = fa[self.t_idx_angles0[m_a0]]
        
        f_a1 = torch.zeros_like(self.t_idx_angles1, dtype=torch.float32)
        m_a1 = self.t_idx_angles1 >= 0
        f_a1[m_a1] = fa[self.t_idx_angles1[m_a1]]

        f_d = torch.zeros_like(self.t_idx_diheds, dtype=torch.float32)
        m_d = self.t_idx_diheds >= 0
        f_d[m_d] = fd[self.t_idx_diheds[m_d]]

        # Concatenate and slice to 52 dimensions as required by params
        f_all = torch.cat([self.t_feature_atypes, f_b, f_a0, f_a1, f_d], dim=-1)
        f_52 = f_all[:, :, :52] # Slice to 52 dimensions
        
        # 3. NN
        f = f_52
        for layer in self.fc0: f = torch.tanh(layer(f))
        
        # Message Passing
        f_center = f[:, 0, :]
        f_nb0 = f[:, 1:4, :]
        f_nb1 = f[:, 4:7, :]
        nbc0 = self.t_nb_connect[:, :3].unsqueeze(-1)
        nbc1 = self.t_nb_connect[:, 3:].unsqueeze(-1)
        
        weighted_nb0 = torch.sum(nbc0 * f_nb0, dim=1)
        weighted_nb1 = torch.sum(nbc1 * f_nb1, dim=1)
        nb0 = torch.sum(self.t_nb_connect[:, :3], dim=1, keepdim=True).clamp(min=1e-5)
        nb1 = torch.sum(self.t_nb_connect[:, 3:], dim=1, keepdim=True).clamp(min=1e-5)
        
        h0 = (torch.sum(self.t_nb_connect[:, :3], dim=1, keepdim=True) > 0).to(pos.dtype)
        h1 = (torch.sum(self.t_nb_connect[:, 3:], dim=1, keepdim=True) > 0).to(pos.dtype)
        
        f = f_center * (1 - h0 * self.w - h1 * self.w) + self.w * weighted_nb0 / nb0 + self.w * weighted_nb1 / nb1
        
        for layer in self.fc1: f = torch.tanh(layer(f))
        energies = self.fc_final(f).squeeze(-1)
        return torch.sum(self.t_weights * energies) * self.sigma + self.mu


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cached_torchscript = OUTPUTS / "dme_10k_vectorized.pt"
    
    all_positions_angstrom, atoms_per_molecule, molecule_count, box_length, atoms, bonds = build_replicated_positions()

    # 1. Build monomer graph
    print("Building monomer graph...")
    monomer_graph = from_pdb(str(INPUTS / "DME.pdb"))
    monomer_graph.get_all_subgraphs(nn=1, typify=True)
    monomer_graph.prepare_subgraph_feature_calc()
    
    # 2. Replicate indices
    print(f"Replicating indices for {molecule_count} molecules...")
    n_atoms_monomer = len(monomer_graph.list_atom_elems)
    
    def to_torch(x):
        if hasattr(x, "tolist"): return torch.tensor(np.array(x))
        return torch.as_tensor(x)

    t_f_a = torch.tile(to_torch(monomer_graph.feature_atypes), (molecule_count, 1, 1)).float()
    t_w = (torch.tile(to_torch(monomer_graph.weights), (molecule_count,)) / molecule_count).float()
    t_n_c = torch.tile(to_torch(monomer_graph.nb_connect), (molecule_count, 1)).float()
    t_b0 = torch.tile(to_torch(monomer_graph.b0), (molecule_count,)).float()
    t_ca0 = torch.tile(to_torch(monomer_graph.cos_a0), (molecule_count,)).float()
    
    def tile_ic_map(idx, n_mols, ic_count):
        idx_t = to_torch(idx); tiled = []
        for i in range(n_mols): tiled.append(torch.where(idx_t >= 0, idx_t + i * ic_count, idx_t))
        return torch.cat(tiled, dim=0).long()

    t_idx_b = tile_ic_map(monomer_graph.feature_indices['bonds'], molecule_count, len(monomer_graph.bonds))
    t_idx_a0 = tile_ic_map(monomer_graph.feature_indices['angles0'], molecule_count, len(monomer_graph.angles))
    t_idx_a1 = tile_ic_map(monomer_graph.feature_indices['angles1'], molecule_count, len(monomer_graph.angles))
    t_idx_d = tile_ic_map(monomer_graph.feature_indices['diheds'], molecule_count, len(monomer_graph.diheds))
    t_b_a = tile_ic_map(torch.tensor(monomer_graph.bonds), molecule_count, n_atoms_monomer)
    t_a_a = tile_ic_map(torch.tensor(monomer_graph.angles), molecule_count, n_atoms_monomer)
    t_d_a = tile_ic_map(torch.tensor(monomer_graph.diheds), molecule_count, n_atoms_monomer)

    # 3. Create Model
    model = sGNNForceOptimized(
        n_layers=(3, 2), sizes=[(40, 20, 20), (20, 10)], n_features=52,
        tiled_bonds_atoms=t_b_a, tiled_b0=t_b0,
        tiled_angles_atoms=t_a_a, tiled_cos_a0=t_ca0,
        tiled_diheds_atoms=t_d_a,
        tiled_idx_bonds=t_idx_b, tiled_idx_angles0=t_idx_a0,
        tiled_idx_angles1=t_idx_a1, tiled_idx_diheds=t_idx_d,
        tiled_feature_atypes=t_f_a, tiled_weights=t_w,
        tiled_nb_connect=t_n_c
    )
    load_sgnn_params(model, INPUTS / "params_sgnn.pickle")
    model.eval()

    print("Scripting model...")
    class FinalWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, p, b): return self.m(p, b) * 4.184
            
    scripted = torch.jit.script(FinalWrapper(model))
    scripted.save(str(cached_torchscript))

    # 4. OpenMM Setup
    system = mm.System()
    for _ in range(TARGET_ATOMS): system.addParticle(1.0 * unit.amu)
    box_nm = box_length / 10.0
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box_nm,0,0)*unit.nanometer, mm.Vec3(0,box_nm,0)*unit.nanometer, mm.Vec3(0,0,box_nm)*unit.nanometer)
    
    force = openmmtorch.TorchForce(str(cached_torchscript))
    force.setUsesPeriodicBoundaryConditions(True)
    system.addForce(force)
    
    integrator = mm.VerletIntegrator(TIMESTEP_FS * unit.femtosecond)
    platform = mm.Platform.getPlatformByName("CUDA")
    context = mm.Context(system, integrator, platform, {"Precision": "single"})
    
    context.setPositions((all_positions_angstrom / 10.0) * unit.nanometer)
    
    print("Warmup...")
    integrator.step(WARMUP_STEPS)
    
    print("Measuring...")
    t0 = time.perf_counter()
    integrator.step(MEASURE_BLOCKS * STEPS_PER_BLOCK)
    total_time = time.perf_counter() - t0
    
    mean_time = total_time / (MEASURE_BLOCKS * STEPS_PER_BLOCK)
    speed = 86400.0 * TIMESTEP_FS / (mean_time * 1e6)
    
    print(f"\nOPTIMIZED CUDA Speed: {speed:.4f} ns/day ({mean_time*1000:.2f} ms/step)")

if __name__ == "__main__":
    main()
