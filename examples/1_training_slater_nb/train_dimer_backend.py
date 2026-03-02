#!/usr/bin/env python
"""Train Slater short-range parameters directly on FF parameter tree."""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, value_and_grad, vmap
from openmm.app import CutoffPeriodic, PDBFile
from openmm.unit import angstrom

from dmff.api import Hamiltonian
from dmff.common import nblist

COMPONENTS = ("ex", "es", "pol", "disp", "dhf", "tot")
COMPONENT_WEIGHTS = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 1.0])
FORCES_WITH_A = (
    "SlaterExForce",
    "SlaterSrEsForce",
    "SlaterSrPolForce",
    "SlaterSrDispForce",
    "SlaterDhfForce",
)
FORCES_WITH_B = (
    "SlaterExForce",
    "SlaterSrEsForce",
    "SlaterSrPolForce",
    "SlaterSrDispForce",
    "SlaterDhfForce",
    "QqTtDampingForce",
    "SlaterDampingForce",
)


class BasePairs:
    def __init__(self, ff: str, pdb: str, pdb_a: str, pdb_b: str):
        pdb_obj = PDBFile(pdb)
        pdb_a_obj = PDBFile(pdb_a)
        pdb_b_obj = PDBFile(pdb_b)

        self.h = Hamiltonian(ff)
        self.pots = self.h.createPotential(
            pdb_obj.topology,
            nonbondedCutoff=25 * angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4,
        )
        self.pots_a = self.h.createPotential(
            pdb_a_obj.topology,
            nonbondedCutoff=25 * angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4,
        )
        self.pots_b = self.h.createPotential(
            pdb_b_obj.topology,
            nonbondedCutoff=25 * angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4,
        )

        self.pos = jnp.array(pdb_obj.positions._value)
        self.pos_a = jnp.array(pdb_a_obj.positions._value)
        self.pos_b = jnp.array(pdb_b_obj.positions._value)

        self.box = jnp.eye(3) * 6
        self.rc = 2.5
        self.nblist = nblist.NeighborList(self.box, self.rc, self.pots.meta["cov_map"])
        self.nblist_a = nblist.NeighborList(self.box, self.rc, self.pots_a.meta["cov_map"])
        self.nblist_b = nblist.NeighborList(self.box, self.rc, self.pots_b.meta["cov_map"])
        self.nblist.allocate(self.pos)
        self.nblist_a.allocate(self.pos_a)
        self.nblist_b.allocate(self.pos_b)

        self.pairs = self.nblist.pairs
        self.pairs_a = self.nblist_a.pairs
        self.pairs_b = self.nblist_b.pairs
        self.pairs_ab = self.pairs[self.pairs[:, 0] < self.pairs[:, 1]]
        self.pairs_a = self.pairs_a[self.pairs_a[:, 0] < self.pairs_a[:, 1]]
        self.pairs_b = self.pairs_b[self.pairs_b[:, 0] < self.pairs_b[:, 1]]

        mapping = {
            "ex": "SlaterExForce",
            "sr_es": "SlaterSrEsForce",
            "sr_pol": "SlaterSrPolForce",
            "sr_disp": "SlaterSrDispForce",
            "dhf": "SlaterDhfForce",
            "dmp_es": "QqTtDampingForce",
            "dmp_disp": "SlaterDampingForce",
        }
        for name, force_name in mapping.items():
            setattr(self, f"pots_{name}", self.pots.dmff_potentials[force_name])
            setattr(self, f"pots_{name}_a", self.pots_a.dmff_potentials[force_name])
            setattr(self, f"pots_{name}_b", self.pots_b.dmff_potentials[force_name])

    def cal_e(self, params_ff: dict[str, dict[str, jnp.ndarray]], pos_a: jnp.ndarray, pos_b: jnp.ndarray):
        pos_a = pos_a * 0.1
        pos_b = pos_b * 0.1
        pos_ab = jnp.concatenate([pos_a, pos_b], axis=0)

        e_ex = self.pots_ex(pos_ab, self.box, self.pairs_ab, params_ff) - self.pots_ex_a(
            pos_a, self.box, self.pairs_a, params_ff
        ) - self.pots_ex_b(pos_b, self.box, self.pairs_b, params_ff)

        e_dmp_es = self.pots_dmp_es(pos_ab, self.box, self.pairs_ab, params_ff) - self.pots_dmp_es_a(
            pos_a, self.box, self.pairs_a, params_ff
        ) - self.pots_dmp_es_b(pos_b, self.box, self.pairs_b, params_ff)
        e_sr_es = self.pots_sr_es(pos_ab, self.box, self.pairs_ab, params_ff) - self.pots_sr_es_a(
            pos_a, self.box, self.pairs_a, params_ff
        ) - self.pots_sr_es_b(pos_b, self.box, self.pairs_b, params_ff)

        e_sr_pol = self.pots_sr_pol(pos_ab, self.box, self.pairs_ab, params_ff) - self.pots_sr_pol_a(
            pos_a, self.box, self.pairs_a, params_ff
        ) - self.pots_sr_pol_b(pos_b, self.box, self.pairs_b, params_ff)

        e_dmp_disp = self.pots_dmp_disp(
            pos_ab, self.box, self.pairs_ab, params_ff
        ) - self.pots_dmp_disp_a(pos_a, self.box, self.pairs_a, params_ff) - self.pots_dmp_disp_b(
            pos_b, self.box, self.pairs_b, params_ff
        )
        e_sr_disp = self.pots_sr_disp(pos_ab, self.box, self.pairs_ab, params_ff) - self.pots_sr_disp_a(
            pos_a, self.box, self.pairs_a, params_ff
        ) - self.pots_sr_disp_b(pos_b, self.box, self.pairs_b, params_ff)

        e_dhf = self.pots_dhf(pos_ab, self.box, self.pairs_ab, params_ff) - self.pots_dhf_a(
            pos_a, self.box, self.pairs_a, params_ff
        ) - self.pots_dhf_b(pos_b, self.box, self.pairs_b, params_ff)

        e_es = e_dmp_es + e_sr_es
        e_pol = e_sr_pol
        e_disp = e_dmp_disp + e_sr_disp
        e_tot = e_ex + e_es + e_pol + e_disp + e_dhf
        return e_ex, e_es, e_pol, e_disp, e_dhf, e_tot


@jit
def calculate_weights(e_tot_full: jnp.ndarray, thresh: float):
    kt = 2.494
    return jnp.where(e_tot_full < thresh, 1.0, jnp.exp(-(e_tot_full - thresh) / kt))


def get_all_contain_key(data: dict[str, Any], arr: list[str]):
    dimer = []
    for key in data:
        a, b = key.split("_")[-2:]
        if a in arr and b in arr:
            dimer.append(key)
    return dimer


def get_all_homo_key(data: dict[str, Any], arr: list[str]):
    dimer = []
    for key in data:
        a, b = key.split("_")[-2:]
        if a == b and b in arr:
            dimer.append(key)
    return dimer


def tree_to_jax(params_tree: dict[str, dict[str, Any]]):
    converted: dict[str, dict[str, jnp.ndarray]] = {}
    for force_name, vals in params_tree.items():
        converted[force_name] = {}
        for key, value in vals.items():
            converted[force_name][key] = jnp.asarray(value)
    return converted


def apply_legacy_restart(
    params_tree: dict[str, dict[str, jnp.ndarray]], legacy_params: dict[str, Any]
) -> dict[str, dict[str, jnp.ndarray]]:
    if "A_ex" in legacy_params:
        params_tree["SlaterExForce"]["A"] = jnp.asarray(legacy_params["A_ex"])
    if "A_es" in legacy_params:
        params_tree["SlaterSrEsForce"]["A"] = jnp.asarray(legacy_params["A_es"])
    if "A_pol" in legacy_params:
        params_tree["SlaterSrPolForce"]["A"] = jnp.asarray(legacy_params["A_pol"])
    if "A_disp" in legacy_params:
        params_tree["SlaterSrDispForce"]["A"] = jnp.asarray(legacy_params["A_disp"])
    if "A_dhf" in legacy_params:
        params_tree["SlaterDhfForce"]["A"] = jnp.asarray(legacy_params["A_dhf"])

    if "B" in legacy_params:
        for force_name in FORCES_WITH_B:
            if force_name in params_tree and "B" in params_tree[force_name]:
                params_tree[force_name]["B"] = jnp.asarray(legacy_params["B"])
    if "Q" in legacy_params and "QqTtDampingForce" in params_tree:
        params_tree["QqTtDampingForce"]["Q"] = jnp.asarray(legacy_params["Q"])
    if "C6" in legacy_params and "SlaterDampingForce" in params_tree:
        params_tree["SlaterDampingForce"]["C6"] = jnp.asarray(legacy_params["C6"])
    if "C8" in legacy_params and "SlaterDampingForce" in params_tree:
        params_tree["SlaterDampingForce"]["C8"] = jnp.asarray(legacy_params["C8"])
    if "C10" in legacy_params and "SlaterDampingForce" in params_tree:
        params_tree["SlaterDampingForce"]["C10"] = jnp.asarray(legacy_params["C10"])
    return params_tree


def get_ff_params(ff: str, restart: str | None, random_scale: float, seed: int):
    base_tree = tree_to_jax(Hamiltonian(ff).getParameters().parameters)
    if restart is None:
        rng = np.random.default_rng(seed)
        for force_name in FORCES_WITH_A:
            base_tree[force_name]["A"] = jnp.asarray(
                rng.random(base_tree[force_name]["A"].shape) * random_scale
            )
        return base_tree

    with open(restart, "rb") as ifile:
        loaded = pickle.load(ifile)

    if hasattr(loaded, "parameters"):
        return tree_to_jax(loaded.parameters)
    if isinstance(loaded, dict):
        if all(k in loaded and isinstance(loaded[k], dict) for k in FORCES_WITH_A):
            return tree_to_jax(loaded)
        return apply_legacy_restart(base_tree, loaded)
    raise ValueError(f"Unsupported restart file format: {type(loaded)}")


def mask_ff_grads(grads: dict[str, dict[str, jnp.ndarray]], train_b: bool):
    masked = {}
    for force_name, vals in grads.items():
        masked[force_name] = {}
        for key, value in vals.items():
            trainable_a = force_name in FORCES_WITH_A and key == "A"
            trainable_b = train_b and force_name in FORCES_WITH_B and key == "B"
            if trainable_a or trainable_b:
                masked[force_name][key] = value
            else:
                masked[force_name][key] = jnp.zeros_like(value)
    return masked


def preprocess_scan_data(scan_data: dict[str, Any]):
    return {
        "posA": jnp.asarray(scan_data["posA"]),
        "posB": jnp.asarray(scan_data["posB"]),
        "wts": jnp.asarray(scan_data["wts"]),
        "ex": jnp.asarray(scan_data["ex"]),
        "es": jnp.asarray(scan_data["es"]),
        "pol": jnp.asarray(scan_data["pol"]),
        "disp": jnp.asarray(scan_data["disp"]),
        "dhf": jnp.asarray(scan_data["dhf"]),
        "tot": jnp.asarray(scan_data["tot"]),
    }


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent.parent
    parser = argparse.ArgumentParser(description="Train Slater dimer parameters on FF param tree")
    parser.add_argument("--data-file", default=str(repo_dir / "data" / "data_dimer.pickle"))
    parser.add_argument("--dimer-bank", default=str(repo_dir / "data" / "dimer_bank"))
    parser.add_argument("--pdb-bank", default=str(repo_dir / "data" / "pdb_bank"))
    parser.add_argument("--ff", default=str(script_dir / "phyneo_ecl.xml"))
    parser.add_argument("--restart", default=None)
    parser.add_argument("--out", default=str(script_dir / "params" / "params_ff_backend.pickle"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--thresh", type=float, default=25.0)
    parser.add_argument("--random-scale", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-species", default="Li,PF6,DMC,EC")
    parser.add_argument("--repulsive-ions", default="Li,Na,PF6,BOB,FSI,TFSI,BF4,DFP,DFOB")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--debug-nans", action="store_true")
    parser.add_argument("--freeze-b", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)

    os.environ["MPLCONFIGDIR"] = str(Path.cwd() / "configs")
    np.random.seed(args.seed)

    with open(args.data_file, "rb") as ifile:
        data = pickle.load(ifile)

    repulsive_ions = [s.strip() for s in args.repulsive_ions.split(",") if s.strip()]
    dimer_repulsive = get_all_homo_key(data, repulsive_ions)
    for pair in data.keys():
        if pair in dimer_repulsive:
            for sid in data[pair].keys():
                npts = len(data[pair][sid]["tot_full"])
                data[pair][sid]["wts"] = jnp.ones(npts)
        else:
            for sid in data[pair].keys():
                data[pair][sid]["wts"] = calculate_weights(data[pair][sid]["tot_full"], args.thresh)

    train_species = [s.strip() for s in args.train_species.split(",") if s.strip()]
    dimer_train = get_all_contain_key(data, train_species)
    dimer_train.sort()
    print(f"Training pairs ({len(dimer_train)}): {dimer_train}")

    params = get_ff_params(args.ff, args.restart, random_scale=args.random_scale, seed=args.seed)

    class_instances = {}
    cal_energy = {}
    for pair in dimer_train:
        conf, numb_conf, monomer_a, monomer_b = pair.split("_")
        _ = conf
        dimer_pdb = Path(args.dimer_bank) / f"dimer_{numb_conf}_{monomer_a}_{monomer_b}.pdb"
        pdb_a = Path(args.pdb_bank) / f"{monomer_a}.pdb"
        pdb_b = Path(args.pdb_bank) / f"{monomer_b}.pdb"
        class_instances[pair] = BasePairs(args.ff, str(dimer_pdb), str(pdb_a), str(pdb_b))
    for class_name, class_instance in class_instances.items():
        cal_energy[class_name] = jit(vmap(class_instance.cal_e, in_axes=(None, 0, 0), out_axes=(0, 0, 0, 0, 0, 0)))

    processed_data = {}
    for key in dimer_train:
        processed_data[key] = {}
        for batch, scan_data in data[key].items():
            processed_data[key][batch] = preprocess_scan_data(scan_data)

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    mse_loss_grad = {}
    train_step = {}
    for key in dimer_train:
        sample_batches = list(data[key].keys())
        if not sample_batches:
            raise ValueError(f"No scan batches found for key: {key}")
        batch = sample_batches[0]

        def mse_loss(params_local, scan_data, pair_key=key):
            weights_pts = scan_data["wts"]
            e_ex, e_es, e_pol, e_disp, e_dhf, e_tot = cal_energy[pair_key](
                params_local, scan_data["posA"], scan_data["posB"]
            )
            pred = jnp.stack([e_ex, e_es, e_pol, e_disp, e_dhf, e_tot], axis=0)
            ref = jnp.stack([scan_data[c] for c in COMPONENTS], axis=0)
            norm = jnp.sum(weights_pts)
            errs = jnp.sum((ref - pred) ** 2 * (weights_pts[None, :] / norm), axis=1)
            return jnp.sum(COMPONENT_WEIGHTS * errs)

        mse_loss_grad[key] = jit(value_and_grad(mse_loss, argnums=(0)))

        @jit
        def train_step_fn(params_local, opt_state_local, scan_data, pair_key=key):
            loss, grads = mse_loss_grad[pair_key](params_local, scan_data)
            masked_grads = mask_ff_grads(grads, train_b=not args.freeze_b)
            updates, new_opt_state = optimizer.update(masked_grads, opt_state_local)
            new_params = optax.apply_updates(params_local, updates)
            return new_params, new_opt_state, loss

        train_step[key] = train_step_fn

        err, _ = mse_loss_grad[key](params, processed_data[key][batch])
        print(f"{key} init loss {float(err):.6f}")

    trunk = []
    for key in dimer_train:
        for batch in data[key]:
            trunk.append((key, batch))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i_epoch in range(args.epochs):
        np.random.shuffle(trunk)
        last_loss = None
        last_key = None
        for key0, batch in trunk:
            params, opt_state, loss = train_step[key0](params, opt_state, processed_data[key0][batch])
            last_loss = loss
            last_key = key0
        if last_loss is not None and last_key is not None and i_epoch % args.log_every == 0:
            print(f"epoch={i_epoch:04d} loss={float(last_loss):.6f} pair={last_key}")
        if i_epoch % args.save_every == 0:
            with open(out_path, "wb") as ofile:
                pickle.dump(params, ofile)

    with open(out_path, "wb") as ofile:
        pickle.dump(params, ofile)

    print(f"Saved FF-tree params to: {out_path}")
    print(f"Elapsed: {(time.time() - t0):.2f}s")


if __name__ == "__main__":
    main()
