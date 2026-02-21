#!/usr/bin/env python
"""Train Slater short-range parameters for dimers.

This script is a CLI version of ``train_dimer.ipynb`` for reproducible runs.
"""

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

    def cal_e(self, params0: dict[str, Any], pos_a: jnp.ndarray, pos_b: jnp.ndarray):
        params = params_convert(params0)
        pos_a = pos_a * 0.1
        pos_b = pos_b * 0.1
        pos_ab = jnp.concatenate([pos_a, pos_b], axis=0)

        e_ex = self.pots_ex(pos_ab, self.box, self.pairs_ab, params) - self.pots_ex_a(
            pos_a, self.box, self.pairs_a, params
        ) - self.pots_ex_b(pos_b, self.box, self.pairs_b, params)

        e_dmp_es = self.pots_dmp_es(pos_ab, self.box, self.pairs_ab, params) - self.pots_dmp_es_a(
            pos_a, self.box, self.pairs_a, params
        ) - self.pots_dmp_es_b(pos_b, self.box, self.pairs_b, params)
        e_sr_es = self.pots_sr_es(pos_ab, self.box, self.pairs_ab, params) - self.pots_sr_es_a(
            pos_a, self.box, self.pairs_a, params
        ) - self.pots_sr_es_b(pos_b, self.box, self.pairs_b, params)

        e_sr_pol = self.pots_sr_pol(pos_ab, self.box, self.pairs_ab, params) - self.pots_sr_pol_a(
            pos_a, self.box, self.pairs_a, params
        ) - self.pots_sr_pol_b(pos_b, self.box, self.pairs_b, params)

        e_dmp_disp = self.pots_dmp_disp(
            pos_ab, self.box, self.pairs_ab, params
        ) - self.pots_dmp_disp_a(pos_a, self.box, self.pairs_a, params) - self.pots_dmp_disp_b(
            pos_b, self.box, self.pairs_b, params
        )
        e_sr_disp = self.pots_sr_disp(pos_ab, self.box, self.pairs_ab, params) - self.pots_sr_disp_a(
            pos_a, self.box, self.pairs_a, params
        ) - self.pots_sr_disp_b(pos_b, self.box, self.pairs_b, params)

        e_dhf = self.pots_dhf(pos_ab, self.box, self.pairs_ab, params) - self.pots_dhf_a(
            pos_a, self.box, self.pairs_a, params
        ) - self.pots_dhf_b(pos_b, self.box, self.pairs_b, params)

        e_es = e_dmp_es + e_sr_es
        e_pol = e_sr_pol
        e_disp = e_dmp_disp + e_sr_disp
        e_tot = e_ex + e_es + e_pol + e_disp + e_dhf
        return e_ex, e_es, e_pol, e_disp, e_dhf, e_tot


@jit
def calculate_weights(e_tot_full: jnp.ndarray, thresh: float):
    kt = 2.494
    return jnp.piecewise(
        e_tot_full,
        [e_tot_full < thresh, e_tot_full >= thresh],
        [lambda x: jnp.array(1.0), lambda x: jnp.exp(-(x - thresh) / kt)],
    )


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


def get_params(restart: str | None, params0: dict[str, Any], random_scale: float):
    comps = ["ex", "es", "pol", "disp", "dhf"]
    if restart is None:
        params = {}
        sr_forces = {
            "ex": "SlaterExForce",
            "es": "SlaterSrEsForce",
            "pol": "SlaterSrPolForce",
            "disp": "SlaterSrDispForce",
            "dhf": "SlaterDhfForce",
        }

        for k in params0["ADMPPmeForce"]:
            params[k] = params0["ADMPPmeForce"][k]
        for k in params0["ADMPDispPmeForce"]:
            params[k] = params0["ADMPDispPmeForce"][k]

        for c in comps:
            for k in params0[sr_forces[c]]:
                if k == "A":
                    params[f"A_{c}"] = params0[sr_forces[c]][k]
                else:
                    params[k] = params0[sr_forces[c]][k]

        for c in comps:
            key = f"A_{c}"
            params[key] = jnp.array(np.random.random(params[key].shape)) * random_scale

        params["Q"] = params0["QqTtDampingForce"]["Q"]
        return params

    with open(restart, "rb") as ifile:
        return pickle.load(ifile)


def params_convert(params: dict[str, Any]):
    params_ex = {}
    params_sr_es = {}
    params_sr_pol = {}
    params_sr_disp = {}
    params_dhf = {}
    params_dmp_es = {}
    params_dmp_disp = {}

    for k in ["B"]:
        params_ex[k] = params[k]
        params_sr_es[k] = params[k]
        params_sr_pol[k] = params[k]
        params_sr_disp[k] = params[k]
        params_dhf[k] = params[k]
        params_dmp_es[k] = params[k]
        params_dmp_disp[k] = params[k]

    if "C" in params:
        params_ex["C"] = params["C"]
    if "D" in params:
        params_ex["D"] = params["D"]

    params_ex["A"] = params["A_ex"]
    params_sr_es["A"] = params["A_es"]
    params_sr_pol["A"] = params["A_pol"]
    params_sr_disp["A"] = params["A_disp"]
    params_dhf["A"] = params["A_dhf"]

    params_dmp_es["Q"] = params["Q"]
    params_dmp_disp["C6"] = params["C6"]
    params_dmp_disp["C8"] = params["C8"]
    params_dmp_disp["C10"] = params["C10"]

    return {
        "SlaterExForce": params_ex,
        "SlaterSrEsForce": params_sr_es,
        "SlaterSrPolForce": params_sr_pol,
        "SlaterSrDispForce": params_sr_disp,
        "SlaterDhfForce": params_dhf,
        "QqTtDampingForce": params_dmp_es,
        "SlaterDampingForce": params_dmp_disp,
    }


def mask_fn(grads: dict[str, Any]):
    for k in grads:
        if not (k.startswith("A_") or k == "B"):
            grads[k] = 0.0
    return grads


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent.parent
    parser = argparse.ArgumentParser(description="Train Slater dimer parameters")
    parser.add_argument("--data-file", default=str(repo_dir / "data" / "data_dimer.pickle"))
    parser.add_argument("--dimer-bank", default=str(repo_dir / "data" / "dimer_bank"))
    parser.add_argument("--pdb-bank", default=str(repo_dir / "data" / "pdb_bank"))
    parser.add_argument("--ff", default=str(script_dir / "phyneo_ecl.xml"))
    parser.add_argument("--restart", default=None)
    parser.add_argument("--out", default=str(script_dir / "params" / "params.pickle"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--thresh", type=float, default=25.0)
    parser.add_argument("--random-scale", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-species", default="Li,PF6,DMC,EC")
    parser.add_argument("--repulsive-ions", default="Li,Na,PF6,BOB,FSI,TFSI,BF4,DFP,DFOB")
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["MPLCONFIGDIR"] = str(Path.cwd() / "configs")
    np.random.seed(args.seed)

    with open(args.data_file, "rb") as ifile:
        data = pickle.load(ifile)

    repulsive_ions = [s.strip() for s in args.repulsive_ions.split(",") if s.strip()]
    dimer_repulsive = get_all_homo_key(data, repulsive_ions)

    for pair in data.keys():
        if pair in dimer_repulsive:
            for sid in data[pair].keys():
                data[pair][sid]["wts"] = jnp.ones(12)
        else:
            for sid in data[pair].keys():
                e_tot_full = data[pair][sid]["tot_full"]
                data[pair][sid]["wts"] = calculate_weights(e_tot_full, args.thresh)

    train_species = [s.strip() for s in args.train_species.split(",") if s.strip()]
    dimer_train = get_all_contain_key(data, train_species)
    dimer_train.sort()
    print(f"Training pairs ({len(dimer_train)}): {dimer_train}")

    params0 = Hamiltonian(args.ff).getParameters()
    params = get_params(args.restart, params0, random_scale=args.random_scale)

    class_instances = {}
    cal_energy = {}
    mse_loss_grad = {}

    for pair in dimer_train:
        conf, numb_conf, monomer_a, monomer_b = pair.split("_")
        _ = conf
        dimer_pdb = Path(args.dimer_bank) / f"dimer_{numb_conf}_{monomer_a}_{monomer_b}.pdb"
        pdb_a = Path(args.pdb_bank) / f"{monomer_a}.pdb"
        pdb_b = Path(args.pdb_bank) / f"{monomer_b}.pdb"
        class_instances[pair] = BasePairs(args.ff, str(dimer_pdb), str(pdb_a), str(pdb_b))

    for class_name, class_instance in class_instances.items():
        cal_energy[class_name] = jit(
            vmap(class_instance.cal_e, in_axes=(None, 0, 0), out_axes=(0, 0, 0, 0, 0, 0))
        )

    for key in dimer_train:
        sample_batches = list(data[key].keys())
        if not sample_batches:
            raise ValueError(f"No scan batches found for key: {key}")
        batch = sample_batches[0]

        def mse_loss(params_local, scan_data, pair_key=key):
            comps = ["ex", "es", "pol", "disp", "dhf", "tot"]
            weights_comps = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 1.0])
            weights_pts = scan_data["wts"]
            npts = len(weights_pts)

            energies = {c: jnp.zeros(npts) for c in comps}
            e_ex, e_es, e_pol, e_disp, e_dhf, e_tot = cal_energy[pair_key](
                params_local, scan_data["posA"], scan_data["posB"]
            )

            for ipt in range(npts):
                energies["ex"] = energies["ex"].at[ipt].set(e_ex[ipt])
                energies["es"] = energies["es"].at[ipt].set(e_es[ipt])
                energies["pol"] = energies["pol"].at[ipt].set(e_pol[ipt])
                energies["disp"] = energies["disp"].at[ipt].set(e_disp[ipt])
                energies["dhf"] = energies["dhf"].at[ipt].set(e_dhf[ipt])
                energies["tot"] = energies["tot"].at[ipt].set(e_tot[ipt])

            errs = jnp.zeros(len(comps))
            norm = jnp.sum(weights_pts)
            for ic, c in enumerate(comps):
                de = scan_data[c] - energies[c]
                mse = de**2 * weights_pts / norm
                errs = errs.at[ic].set(jnp.sum(mse))
            return jnp.sum(weights_comps * errs)

        mse_loss_grad[key] = jit(value_and_grad(mse_loss, argnums=(0)))
        err, _ = mse_loss_grad[key](params, data[key][batch])
        print(f"{key} init loss {float(err):.6f}")

    trunk = []
    for key in dimer_train:
        for batch in data[key]:
            trunk.append((key, batch))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    t0 = time.time()
    for i_epoch in range(args.epochs):
        np.random.shuffle(trunk)
        last_loss = None
        last_key = None
        for key0, batch in trunk:
            loss, grads = mse_loss_grad[key0](params, data[key0][batch])
            grad = mask_fn(grads)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            last_loss = loss
            last_key = key0

        if last_loss is not None and last_key is not None:
            print(f"epoch={i_epoch:04d} loss={float(last_loss):.6f} pair={last_key}")

        if i_epoch % args.save_every == 0:
            with open(out_path, "wb") as ofile:
                pickle.dump(params, ofile)

    with open(out_path, "wb") as ofile:
        pickle.dump(params, ofile)

    print(f"Saved params to: {out_path}")
    print(f"Elapsed: {(time.time() - t0):.2f}s")


if __name__ == "__main__":
    # Keep behavior close to notebook defaults.
    jax.config.update("jax_debug_nans", True)
    main()
