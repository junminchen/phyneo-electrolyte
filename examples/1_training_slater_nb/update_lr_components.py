#!/usr/bin/env python
"""Recompute and refresh long-range (LR) components in dimer training data."""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from openmm.app import CutoffPeriodic, PDBFile
from openmm.unit import angstrom

from dmff.api import Hamiltonian
from dmff.common import nblist


class BasePairs:
    """Pair-specific energy evaluator for LR electrostatics/polarization/dispersion."""

    def __init__(self, ff: str, dimer_pdb: str, pdb_a: str, pdb_b: str):
        pdb = PDBFile(dimer_pdb)
        pdb_a_obj = PDBFile(pdb_a)
        pdb_b_obj = PDBFile(pdb_b)

        self.h = Hamiltonian(ff)
        self.h_a = Hamiltonian(ff)
        self.h_b = Hamiltonian(ff)

        self.pots = self.h.createPotential(
            pdb.topology,
            nonbondedCutoff=25 * angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4,
            step_pol=20,
        )
        self.pots_a = self.h_a.createPotential(
            pdb_a_obj.topology,
            nonbondedCutoff=25 * angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4,
            step_pol=20,
        )
        self.pots_b = self.h_b.createPotential(
            pdb_b_obj.topology,
            nonbondedCutoff=25 * angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4,
            step_pol=20,
        )

        self.generators = self.h.getGenerators()
        self.generators_a = self.h_a.getGenerators()
        self.generators_b = self.h_b.getGenerators()

        self.pos = jnp.array(pdb.positions._value)
        self.pos_a = jnp.array(pdb_a_obj.positions._value)
        self.pos_b = jnp.array(pdb_b_obj.positions._value)

        self.box = jnp.eye(3) * 6
        rc = 2.5
        self.nblist = nblist.NeighborList(self.box, rc, self.pots.meta["cov_map"])
        self.nblist_a = nblist.NeighborList(self.box, rc, self.pots_a.meta["cov_map"])
        self.nblist_b = nblist.NeighborList(self.box, rc, self.pots_b.meta["cov_map"])
        self.nblist.allocate(self.pos)
        self.nblist_a.allocate(self.pos_a)
        self.nblist_b.allocate(self.pos_b)

        self.pairs = self.nblist.pairs
        self.pairs_a = self.nblist_a.pairs
        self.pairs_b = self.nblist_b.pairs
        self.pairs_ab = self.pairs[self.pairs[:, 0] < self.pairs[:, 1]]
        self.pairs_a = self.pairs_a[self.pairs_a[:, 0] < self.pairs_a[:, 1]]
        self.pairs_b = self.pairs_b[self.pairs_b[:, 0] < self.pairs_b[:, 1]]

        self.pots_es = self.pots.dmff_potentials["ADMPPmeForce"]
        self.pots_es_a = self.pots_a.dmff_potentials["ADMPPmeForce"]
        self.pots_es_b = self.pots_b.dmff_potentials["ADMPPmeForce"]
        self.pots_disp = self.pots.dmff_potentials["ADMPDispPmeForce"]
        self.pots_disp_a = self.pots_a.dmff_potentials["ADMPDispPmeForce"]
        self.pots_disp_b = self.pots_b.dmff_potentials["ADMPDispPmeForce"]

    def cal_e(self, params: Any, pos_a: jnp.ndarray, pos_b: jnp.ndarray):
        pos_a_nm = pos_a * 0.1
        pos_b_nm = pos_b * 0.1
        pos_ab_nm = jnp.concatenate([pos_a_nm, pos_b_nm], axis=0)
        box_nm = self.box

        e_espol_a = self.pots_es_a(pos_a_nm, box_nm, self.pairs_a, params)
        e_espol_b = self.pots_es_b(pos_b_nm, box_nm, self.pairs_b, params)
        e_espol = self.pots_es(pos_ab_nm, box_nm, self.pairs_ab, params) - e_espol_a - e_espol_b

        pme_generator_ab = self.generators[0]
        pme_generator_a = self.generators_a[0]
        pme_generator_b = self.generators_b[0]
        u_ind_ab = jnp.vstack((pme_generator_a.pme_force.U_ind, pme_generator_b.pme_force.U_ind))

        params_pme = params["ADMPPmeForce"]
        map_atypes = self.pots.meta["ADMPPmeForce_map_atomtype"]
        map_poltypes = self.pots.meta["ADMPPmeForce_map_poltype"]
        q_local = params_pme["Q_local"][map_atypes]
        pol = params_pme["pol"][map_poltypes]
        tholes = params_pme["thole"][map_poltypes]
        pme_force = pme_generator_ab.pme_force
        e_nonpol_ab = pme_force.energy_fn(
            pos_ab_nm * 10,
            box_nm * 10,
            self.pairs_ab,
            q_local,
            u_ind_ab,
            pol,
            tholes,
            pme_generator_ab.mScales,
            pme_generator_ab.pScales,
            pme_generator_ab.dScales,
        )
        e_es = e_nonpol_ab - e_espol_a - e_espol_b
        e_pol = e_espol - e_es

        e_disp = (
            self.pots_disp(pos_ab_nm, box_nm, self.pairs_ab, params)
            - self.pots_disp_a(pos_a_nm, box_nm, self.pairs_a, params)
            - self.pots_disp_b(pos_b_nm, box_nm, self.pairs_b, params)
        )
        return e_es, e_pol, e_disp


def get_all_contain_key(data: dict[str, Any], species: list[str]):
    selected = []
    for key in data:
        a, b = key.split("_")[-2:]
        if a in species or b in species:
            selected.append(key)
    return selected


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent.parent
    parser = argparse.ArgumentParser(description="Refresh LR terms in dimer training pickle")
    parser.add_argument("--data-file", required=True, help="Input dimer pickle")
    parser.add_argument("--out", required=True, help="Output pickle path")
    parser.add_argument("--ff", default=str(script_dir / "phyneo_ecl.xml"))
    parser.add_argument("--dimer-bank", default=str(repo_dir / "data" / "dimer_bank"))
    parser.add_argument("--pdb-bank", default=str(repo_dir / "data" / "pdb_bank"))
    parser.add_argument(
        "--pairs",
        default="",
        help="Comma-separated pair keys to process, e.g. conf_051_Li_PF6,conf_062_Li_EC",
    )
    parser.add_argument(
        "--species",
        default="",
        help="Comma-separated species filter (select keys that contain any species)",
    )
    parser.add_argument("--batch", default="", help="Only process one batch id (default: all batches)")
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit processed pair count (0 means all)")
    parser.add_argument("--dry-run", action="store_true", help="Compute and report only; do not write output")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="If output exists, save backup as <out>.bak before overwrite",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        with open(args.data_file, "rb") as ifile:
            data = pickle.load(ifile)
    except Exception as exc:
        header = Path(args.data_file).read_text(errors="ignore").splitlines()[:1]
        if header and header[0].startswith("version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"{args.data_file} is a Git LFS pointer, not real pickle content. "
                "Run `git lfs pull` to fetch dataset blobs first."
            ) from exc
        raise

    params = Hamiltonian(args.ff).getParameters()

    explicit_pairs = [x.strip() for x in args.pairs.split(",") if x.strip()]
    species = [x.strip() for x in args.species.split(",") if x.strip()]

    if explicit_pairs:
        dimer_keys = [k for k in explicit_pairs if k in data]
        missing = [k for k in explicit_pairs if k not in data]
        if missing:
            print(f"Warning: {len(missing)} pairs not found and skipped: {missing}")
    elif species:
        dimer_keys = sorted(get_all_contain_key(data, species))
    else:
        dimer_keys = sorted(data.keys())

    if args.max_pairs > 0:
        dimer_keys = dimer_keys[: args.max_pairs]

    print(f"Selected pairs: {len(dimer_keys)}")
    if not dimer_keys:
        raise ValueError("No pairs selected.")

    class_instances: dict[str, BasePairs] = {}
    cal_energy = {}
    for pair in dimer_keys:
        _, numb_conf, monomer_a, monomer_b = pair.split("_")
        dimer_file = Path(args.dimer_bank) / f"dimer_{numb_conf}_{monomer_a}_{monomer_b}.pdb"
        pdb_a_file = Path(args.pdb_bank) / f"{monomer_a}.pdb"
        pdb_b_file = Path(args.pdb_bank) / f"{monomer_b}.pdb"
        class_instances[pair] = BasePairs(args.ff, str(dimer_file), str(pdb_a_file), str(pdb_b_file))
        cal_energy[pair] = jit(vmap(class_instances[pair].cal_e, in_axes=(None, 0, 0), out_axes=(0, 0, 0)))

    n_pairs = 0
    n_batches = 0
    n_points = 0
    for key in dimer_keys:
        batches = [args.batch] if args.batch else list(data[key].keys())
        for sid in batches:
            if sid not in data[key]:
                print(f"Warning: batch '{sid}' not found for key '{key}', skipped.")
                continue

            scan_res = data[key][sid]
            if "tot_full" not in scan_res:
                scan_res["tot_full"] = np.array(scan_res["tot"], copy=True)

            pos_a = jnp.asarray(scan_res["posA"])
            pos_b = jnp.asarray(scan_res["posB"])
            e_es, e_pol, e_disp = cal_energy[key](params, pos_a, pos_b)

            # Existing behavior from notebook: restore old LR first if present, then replace.
            if "lr_tot" in scan_res:
                scan_res["tot"] = np.asarray(scan_res["tot"]) + np.asarray(scan_res["lr_tot"])
                scan_res["es"] = np.asarray(scan_res["es"]) + np.asarray(scan_res["lr_es"])
                scan_res["pol"] = np.asarray(scan_res["pol"]) + np.asarray(scan_res["lr_pol"])
                scan_res["disp"] = np.asarray(scan_res["disp"]) + np.asarray(scan_res["lr_disp"])

            scan_res["lr_es"] = np.asarray(e_es)
            scan_res["lr_pol"] = np.asarray(e_pol)
            scan_res["lr_disp"] = np.asarray(e_disp)
            scan_res["lr_tot"] = np.asarray(e_es + e_pol + e_disp)

            scan_res["tot"] = np.asarray(scan_res["tot"]) - scan_res["lr_tot"]
            scan_res["es"] = np.asarray(scan_res["es"]) - scan_res["lr_es"]
            scan_res["pol"] = np.asarray(scan_res["pol"]) - scan_res["lr_pol"]
            scan_res["disp"] = np.asarray(scan_res["disp"]) - scan_res["lr_disp"]

            n_batches += 1
            n_points += len(scan_res["tot"])
        n_pairs += 1

    print(f"Processed pairs={n_pairs}, batches={n_batches}, points={n_points}")
    if args.dry_run:
        print("Dry-run enabled: output file not written.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.backup and out_path.exists():
        backup = out_path.with_suffix(out_path.suffix + ".bak")
        shutil.copy2(out_path, backup)
        print(f"Backup created: {backup}")

    with open(out_path, "wb") as ofile:
        pickle.dump(data, ofile)
    print(f"Saved updated data to: {out_path}")


if __name__ == "__main__":
    main()
