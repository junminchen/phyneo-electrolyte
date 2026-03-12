#!/usr/bin/env python3
"""Minimal reproducer for MPIDForce + openmmtorch incompatibility.

This script deliberately avoids the repository's sGNN model code.  It only:

1. loads the MPID force from `phyneo_ecl.xml`
2. optionally adds a trivial TorchForce that always returns zero energy
3. tries to create an OpenMM Context

If `MPIDForce` alone works but `MPIDForce + zero TorchForce` fails, the issue is
in the plugin/kernel combination rather than in the model implementation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import openmm as mm
from openmm import app, unit
import openmmtorch
import torch
import torch.nn as nn
import mpidplugin


class ZeroTorchForce(nn.Module):
    def forward(self, positions_nm: torch.Tensor, boxvectors_nm: torch.Tensor) -> torch.Tensor:
        del positions_nm, boxvectors_nm
        return torch.zeros((), dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal reproducer for MPIDForce + openmmtorch conflicts"
    )
    parser.add_argument("--pdb", default="init.pdb", help="PDB file relative to this script")
    parser.add_argument("--xml", default="phyneo_ecl.xml", help="Force-field XML relative to this script")
    parser.add_argument(
        "--platform",
        default="CUDA",
        help="OpenMM platform to test, for example CUDA, CPU, or Reference",
    )
    parser.add_argument(
        "--with-torch-force",
        action="store_true",
        help="Add a trivial zero-energy TorchForce on top of MPIDForce",
    )
    parser.add_argument(
        "--cuda-precision",
        default="mixed",
        help="CUDA precision mode when using the CUDA platform",
    )
    return parser.parse_args()


def resolve_local(path_str: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_dir / path


def build_zero_torch_force() -> tuple[openmmtorch.TorchForce, str]:
    model = ZeroTorchForce().eval()
    positions = torch.zeros((1, 3), dtype=torch.float32)
    box = torch.eye(3, dtype=torch.float32)
    traced = torch.jit.trace(model, (positions, box))
    fd, temp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    traced.save(temp_path)

    force = openmmtorch.TorchForce(temp_path)
    force.setUsesPeriodicBoundaryConditions(True)
    force.setForceGroup(1)
    return force, temp_path


def describe_force(force) -> str:
    if mpidplugin.MPIDForce.isinstance(force):
        return "MPIDForce"
    return force.__class__.__name__


def main() -> int:
    args = parse_args()
    pdb_path = resolve_local(args.pdb)
    xml_path = resolve_local(args.xml)

    print(f"Testing platform: {args.platform}")
    print(f"PDB: {pdb_path}")
    print(f"XML: {xml_path}")
    print(f"Add TorchForce: {args.with_torch_force}")

    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(str(xml_path))
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.6 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=True,
    )

    temp_path: str | None = None
    try:
        if args.with_torch_force:
            torch_force, temp_path = build_zero_torch_force()
            system.addForce(torch_force)

        print("Forces in system:")
        for i in range(system.getNumForces()):
            print(f"  {i}: {describe_force(system.getForce(i))}")

        platform = mm.Platform.getPlatformByName(args.platform)
        properties = {}
        if args.platform.upper() == "CUDA":
            properties["CudaPrecision"] = args.cuda_precision

        integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
        context = mm.Context(system, integrator, platform, properties)
        context.setPositions(pdb.positions)
        energy = context.getState(getEnergy=True).getPotentialEnergy()

        print("Context creation: OK")
        print(f"Potential energy: {energy}")
        return 0
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    raise SystemExit(main())
