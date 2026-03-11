#!/usr/bin/env python
"""Plot distance vs energy (reference vs prediction) for one scan batch."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, vmap
from openmm.app import CutoffPeriodic, PDBFile
from openmm.unit import angstrom

from dmff.api import Hamiltonian
from dmff.common import nblist

COMPONENTS = ("ex", "es", "pol", "disp", "dhf", "tot")
COMPONENT_COLORS = {
    "ex": "tab:red",
    "es": "tab:blue",
    "pol": "tab:green",
    "disp": "tab:orange",
    "dhf": "tab:purple",
    "tot": "black",
}


# ── reuse BasePairs from train_dimer_backend ──────────────────────────────────

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

        self.pairs_ab = self.nblist.pairs
        self.pairs_a = self.nblist_a.pairs
        self.pairs_b = self.nblist_b.pairs
        self.pairs_ab = self.pairs_ab[self.pairs_ab[:, 0] < self.pairs_ab[:, 1]]
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

    def cal_e(self, params_ff, pos_a, pos_b):
        pos_a = pos_a * 0.1
        pos_b = pos_b * 0.1
        pos_ab = jnp.concatenate([pos_a, pos_b], axis=0)

        e_ex = (self.pots_ex(pos_ab, self.box, self.pairs_ab, params_ff)
                - self.pots_ex_a(pos_a, self.box, self.pairs_a, params_ff)
                - self.pots_ex_b(pos_b, self.box, self.pairs_b, params_ff))
        e_dmp_es = (self.pots_dmp_es(pos_ab, self.box, self.pairs_ab, params_ff)
                    - self.pots_dmp_es_a(pos_a, self.box, self.pairs_a, params_ff)
                    - self.pots_dmp_es_b(pos_b, self.box, self.pairs_b, params_ff))
        e_sr_es = (self.pots_sr_es(pos_ab, self.box, self.pairs_ab, params_ff)
                   - self.pots_sr_es_a(pos_a, self.box, self.pairs_a, params_ff)
                   - self.pots_sr_es_b(pos_b, self.box, self.pairs_b, params_ff))
        e_sr_pol = (self.pots_sr_pol(pos_ab, self.box, self.pairs_ab, params_ff)
                    - self.pots_sr_pol_a(pos_a, self.box, self.pairs_a, params_ff)
                    - self.pots_sr_pol_b(pos_b, self.box, self.pairs_b, params_ff))
        e_dmp_disp = (self.pots_dmp_disp(pos_ab, self.box, self.pairs_ab, params_ff)
                      - self.pots_dmp_disp_a(pos_a, self.box, self.pairs_a, params_ff)
                      - self.pots_dmp_disp_b(pos_b, self.box, self.pairs_b, params_ff))
        e_sr_disp = (self.pots_sr_disp(pos_ab, self.box, self.pairs_ab, params_ff)
                     - self.pots_sr_disp_a(pos_a, self.box, self.pairs_a, params_ff)
                     - self.pots_sr_disp_b(pos_b, self.box, self.pairs_b, params_ff))
        e_dhf = (self.pots_dhf(pos_ab, self.box, self.pairs_ab, params_ff)
                 - self.pots_dhf_a(pos_a, self.box, self.pairs_a, params_ff)
                 - self.pots_dhf_b(pos_b, self.box, self.pairs_b, params_ff))

        e_es = e_dmp_es + e_sr_es
        e_pol = e_sr_pol
        e_disp = e_dmp_disp + e_sr_disp
        e_tot = e_ex + e_es + e_pol + e_disp + e_dhf
        return e_ex, e_es, e_pol, e_disp, e_dhf, e_tot


# ── helpers ───────────────────────────────────────────────────────────────────

def min_intermolecular_dist(posA: np.ndarray, posB: np.ndarray) -> np.ndarray:
    """Minimum atom-atom distance between monomer A and B for each frame.

    Args:
        posA: (n_frames, n_atoms_a, 3) in Angstrom
        posB: (n_frames, n_atoms_b, 3) in Angstrom
    Returns:
        (n_frames,) minimum distances in Angstrom
    """
    dists = []
    for i in range(posA.shape[0]):
        d = np.linalg.norm(posA[i, :, None, :] - posB[i, None, :, :], axis=-1)
        dists.append(d.min())
    return np.array(dists)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent.parent
    parser = argparse.ArgumentParser(description="Plot distance vs energy: ref vs pred")
    parser.add_argument("--data-file", default=str(repo_dir / "data" / "data_dimer.pickle"))
    parser.add_argument("--dimer-bank", default=str(repo_dir / "data" / "dimer_bank"))
    parser.add_argument("--pdb-bank", default=str(repo_dir / "data" / "pdb_bank"))
    parser.add_argument("--ff", default=str(script_dir / "phyneo_ecl.xml"))
    parser.add_argument("--params", default=str(script_dir / "params" / "params_ff_balanced.pickle"))
    parser.add_argument("--pair", default=None, help="Pair key, e.g. conf_001_DMC_DMC. Default: first pair.")
    parser.add_argument("--batch", default=None, help="Batch key, e.g. 000. Default: first batch.")
    parser.add_argument("--out", default=str(script_dir / "output" / "energy_distance.png"))
    parser.add_argument("--components", default="ex,es,pol,disp,dhf,tot",
                        help="Comma-separated components to plot")
    parser.add_argument("--emax", type=float, default=None, help="Y-axis upper limit (kcal/mol)")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Plot all pairs; saves one PNG per pair in the output directory")
    parser.add_argument("--all-batches", action="store_true",
                        help="Aggregate all batches for each pair into one full dimer scan plot")
    return parser.parse_args()


def plot_all_in_one(pair: str, distances: np.ndarray, ref: dict, pred: dict,
                    components: list[str], emax: float | None, out_path: Path):
    """Draw all selected components in a single combined plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f"{pair} — All Components Comparison", fontsize=13, fontweight="bold")
    
    for comp in components:
        color = COMPONENT_COLORS.get(comp, "gray")
        # Reference: solid line with dots
        ax.plot(distances, ref[comp], 'o-', color=color, label=f"Ref {comp}", 
                linewidth=1.5, markersize=3, alpha=0.8)
        # Prediction: dashed line
        ax.plot(distances, pred[comp], '--', color=color, label=f"Pred {comp}", 
                linewidth=2.0, alpha=0.6)

    ax.set_xlabel("Min intermolecular distance (Å)", fontsize=11)
    ax.set_ylabel("Energy (kcal/mol)", fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Focus on the relevant energy range (ignore extreme repulsion if needed)
    if emax is not None:
        ax.set_ylim(top=emax)
    else:
        # Default zoom to see attractive regions clearly
        ax.set_ylim(-20, 50) 

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Combined Plot: {out_path}")


def plot_scan(pair: str, distances: np.ndarray, ref: dict, pred: dict,
              components: list[str], emax: float | None, out_path: Path,
              n_batches: int):
    """Draw and save a full dimer scan (all batches aggregated) figure."""
    ncols = 3
    nrows = (len(components) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"{pair}  —  full scan  ({n_batches} batches, {len(distances)} pts)",
                 fontsize=13, fontweight="bold")

    for idx, comp in enumerate(components):
        ax = axes[idx // ncols][idx % ncols]
        color = COMPONENT_COLORS.get(comp, "gray")
        ax.scatter(distances, ref[comp], s=10, color=color, label="Ref", alpha=0.6, zorder=3)
        ax.scatter(distances, pred[comp], s=10, marker="s", color=color,
                   label="Pred", alpha=0.4, zorder=2)
        ax.set_xlabel("Min intermolecular distance (Å)", fontsize=10)
        ax.set_ylabel("Energy (kcal/mol)", fontsize=10)
        ax.set_title(comp, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if emax is not None:
            ax.set_ylim(top=emax)

    for idx in range(len(components), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_one(pair: str, batch: str, scan: dict, distances: np.ndarray,
             ref: dict, pred: dict, components: list[str],
             emax: float | None, out_path: Path):
    """Draw and save one distance-vs-energy figure."""
    ncols = 3
    nrows = (len(components) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"{pair}  /  batch {batch}", fontsize=13, fontweight="bold")

    for idx, comp in enumerate(components):
        ax = axes[idx // ncols][idx % ncols]
        color = COMPONENT_COLORS.get(comp, "gray")
        ax.plot(distances, ref[comp], "o-", color=color, label="Ref", linewidth=1.5, markersize=4)
        ax.plot(distances, pred[comp], "s--", color=color, label="Pred",
                linewidth=1.5, markersize=4, alpha=0.75)
        ax.set_xlabel("Min intermolecular distance (Å)", fontsize=10)
        ax.set_ylabel("Energy (kcal/mol)", fontsize=10)
        ax.set_title(comp, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if emax is not None:
            ax.set_ylim(top=emax)

    for idx in range(len(components), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def run_pair_all_batches(pair: str, data: dict, params, args) -> None:
    """Aggregate all batches and plot the full dimer scan for one pair."""
    _, numb_conf, monomer_a, monomer_b = pair.split("_")
    dimer_pdb = Path(args.dimer_bank) / f"dimer_{numb_conf}_{monomer_a}_{monomer_b}.pdb"
    pdb_a = Path(args.pdb_bank) / f"{monomer_a}.pdb"
    pdb_b = Path(args.pdb_bank) / f"{monomer_b}.pdb"

    bp = BasePairs(args.ff, str(dimer_pdb), str(pdb_a), str(pdb_b))
    cal_e_batch = jit(vmap(bp.cal_e, in_axes=(None, 0, 0), out_axes=(0, 0, 0, 0, 0, 0)))

    all_dist, all_ref, all_pred = [], {c: [] for c in COMPONENTS}, {c: [] for c in COMPONENTS}

    for batch in sorted(data[pair].keys()):
        scan = data[pair][batch]
        posA = np.array(scan["posA"])
        posB = np.array(scan["posB"])
        dists = min_intermolecular_dist(posA, posB)
        all_dist.append(dists)
        for c in COMPONENTS:
            all_ref[c].append(np.array(scan[c]))
        pred_raw = cal_e_batch(params, jnp.asarray(posA), jnp.asarray(posB))
        for i, c in enumerate(COMPONENTS):
            all_pred[c].append(np.array(pred_raw[i]))

    distances = np.concatenate(all_dist)
    sort_idx = np.argsort(distances)
    distances = distances[sort_idx]
    ref  = {c: np.concatenate(all_ref[c])[sort_idx]  for c in COMPONENTS}
    pred = {c: np.concatenate(all_pred[c])[sort_idx] for c in COMPONENTS}

    components = [c.strip() for c in args.components.split(",")]
    out_dir = Path(args.out).parent
    out_path = out_dir / f"scan_{pair}.png"
    plot_scan(pair, distances, ref, pred, components, args.emax, out_path,
              n_batches=len(data[pair]))
    
    # Also save the combined plot
    plot_all_in_one(pair, distances, ref, pred, components, args.emax,
                     out_dir / f"combined_{pair}.png")


def run_pair(pair: str, batch: str, data: dict, params, args) -> None:
    scan = data[pair][batch]
    posA = np.array(scan["posA"])
    posB = np.array(scan["posB"])

    distances = min_intermolecular_dist(posA, posB)
    sort_idx = np.argsort(distances)
    distances = distances[sort_idx]
    ref = {c: np.array(scan[c])[sort_idx] for c in COMPONENTS}

    _, numb_conf, monomer_a, monomer_b = pair.split("_")
    dimer_pdb = Path(args.dimer_bank) / f"dimer_{numb_conf}_{monomer_a}_{monomer_b}.pdb"
    pdb_a = Path(args.pdb_bank) / f"{monomer_a}.pdb"
    pdb_b = Path(args.pdb_bank) / f"{monomer_b}.pdb"

    bp = BasePairs(args.ff, str(dimer_pdb), str(pdb_a), str(pdb_b))
    cal_e_batch = jit(vmap(bp.cal_e, in_axes=(None, 0, 0), out_axes=(0, 0, 0, 0, 0, 0)))

    posA_jnp = jnp.asarray(posA)
    posB_jnp = jnp.asarray(posB)
    pred_raw = cal_e_batch(params, posA_jnp, posB_jnp)
    pred = {c: np.array(pred_raw[i])[sort_idx] for i, c in enumerate(COMPONENTS)}

    components = [c.strip() for c in args.components.split(",")]
    base_out = Path(args.out).parent / f"energy_distance_{pair}_b{batch}.png" \
               if args.all_pairs else Path(args.out)
    plot_one(pair, batch, scan, distances, ref, pred, components, args.emax, base_out)
    
    # Combined plot
    comb_out = base_out.parent / f"combined_{pair}_b{batch}.png"
    plot_all_in_one(pair, distances, ref, pred, components, args.emax, comb_out)


def main():
    args = parse_args()

    with open(args.data_file, "rb") as f:
        data = pickle.load(f)

    with open(args.params, "rb") as f:
        params = pickle.load(f)

    if args.all_pairs or args.all_batches:
        pairs = sorted(data.keys())
        print(f"Plotting {len(pairs)} pairs ...")
        for pair in pairs:
            print(f"  {pair}")
            if args.all_batches:
                run_pair_all_batches(pair, data, params, args)
            else:
                batch = args.batch or sorted(data[pair].keys())[0]
                run_pair(pair, batch, data, params, args)
    else:
        pair = args.pair or sorted(data.keys())[0]
        if pair not in data:
            raise ValueError(f"Pair '{pair}' not found. Available: {list(data.keys())}")
        batch = args.batch or sorted(data[pair].keys())[0]
        if batch not in data[pair]:
            raise ValueError(f"Batch '{batch}' not found. Available: {list(data[pair].keys())}")
        print(f"Pair: {pair}  Batch: {batch}  Frames: {np.array(data[pair][batch]['posA']).shape[0]}")
        run_pair(pair, batch, data, params, args)


if __name__ == "__main__":
    main()
