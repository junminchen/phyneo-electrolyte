#!/usr/bin/env python
"""Refresh long-range (LR) components in dimer data with strict validation/reporting.

Exit codes:
  0: success
  2: quality checks failed in strict mode
  3: input/path/selection validation error
  4: runtime energy computation error
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from openmm.app import CutoffPeriodic, PDBFile
from openmm.unit import angstrom

from dmff.api import Hamiltonian
from dmff.common import nblist

EXIT_OK = 0
EXIT_CHECK_FAILED = 2
EXIT_INPUT_ERROR = 3
EXIT_RUNTIME_ERROR = 4

REQUIRED_FIELDS = ("posA", "posB", "tot", "es", "pol", "disp")
REQUIRED_LR_FIELDS = ("lr_es", "lr_pol", "lr_disp", "lr_tot")


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_path: Path
    ff: Path
    dimer_bank: Path
    pdb_bank: Path
    pairs: list[str]
    species: list[str]
    batch: str | None
    max_pairs: int
    sample_limit: int
    strict: bool
    fail_fast: bool
    dry_run: bool
    backup: str
    report_json: Path | None
    consistency_atol: float
    consistency_rtol: float


class BasePairs:
    """Pair-specific LR evaluator for electrostatics, polarization, and dispersion."""

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


def parse_args() -> Config:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="Refresh LR terms in dimer training pickle")
    parser.add_argument("--input", required=True, help="Input dimer pickle path")
    parser.add_argument("--output", required=True, help="Output pickle path")
    parser.add_argument("--ff", default=str(script_dir / "phyneo_ecl.xml"), help="Force-field XML")
    parser.add_argument("--dimer-bank", default=str(repo_dir / "data" / "dimer_bank"))
    parser.add_argument("--pdb-bank", default=str(repo_dir / "data" / "pdb_bank"))
    parser.add_argument("--pairs", default="", help="Comma-separated explicit pair keys")
    parser.add_argument("--species", default="", help="Comma-separated species filter")
    parser.add_argument("--batch", default="", help="Only process one batch id")
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit selected pairs (0=all)")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Limit processed (pair,batch) units after selection (0=all)",
    )
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--backup",
        choices=("none", "overwrite", "versioned"),
        default="versioned",
        help="Backup strategy when output exists",
    )
    parser.add_argument("--report-json", default="", help="Optional JSON report output path")
    parser.add_argument("--consistency-atol", type=float, default=1e-6)
    parser.add_argument("--consistency-rtol", type=float, default=1e-6)

    args = parser.parse_args()
    return Config(
        input_path=Path(args.input),
        output_path=Path(args.output),
        ff=Path(args.ff),
        dimer_bank=Path(args.dimer_bank),
        pdb_bank=Path(args.pdb_bank),
        pairs=[x.strip() for x in args.pairs.split(",") if x.strip()],
        species=[x.strip() for x in args.species.split(",") if x.strip()],
        batch=args.batch.strip() or None,
        max_pairs=args.max_pairs,
        sample_limit=args.sample_limit,
        strict=args.strict,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run,
        backup=args.backup,
        report_json=Path(args.report_json) if args.report_json.strip() else None,
        consistency_atol=args.consistency_atol,
        consistency_rtol=args.consistency_rtol,
    )


def is_lfs_pointer(path: Path) -> bool:
    try:
        first = path.read_text(errors="ignore").splitlines()[:1]
    except Exception:
        return False
    return bool(first and first[0].startswith("version https://git-lfs.github.com/spec/v1"))


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    try:
        with open(path, "rb") as ifile:
            return pickle.load(ifile)
    except Exception as exc:
        if is_lfs_pointer(path):
            raise RuntimeError(
                f"{path} is a Git LFS pointer, not real pickle content. "
                "Run `git lfs pull` to fetch dataset blobs first."
            ) from exc
        raise


def choose_pairs(data: dict[str, Any], cfg: Config) -> tuple[list[str], list[str]]:
    warnings: list[str] = []

    if cfg.pairs:
        selected = [k for k in cfg.pairs if k in data]
        missing = [k for k in cfg.pairs if k not in data]
        if missing:
            warnings.append(f"{len(missing)} explicit pairs not found: {missing}")
    elif cfg.species:
        selected = []
        for key in data:
            a, b = key.split("_")[-2:]
            if a in cfg.species or b in cfg.species:
                selected.append(key)
        selected.sort()
    else:
        selected = sorted(data.keys())

    if cfg.max_pairs > 0:
        selected = selected[: cfg.max_pairs]
    return selected, warnings


def add_failure(report: dict[str, Any], pair: str, batch: str, reason: str, detail: str):
    report["checks"]["failures"].append(
        {
            "pair": pair,
            "batch": batch,
            "reason": reason,
            "detail": detail,
        }
    )


def validate_scan_input(scan_res: dict[str, Any]) -> tuple[bool, str]:
    missing = [f for f in REQUIRED_FIELDS if f not in scan_res]
    if missing:
        return False, f"missing required fields: {missing}"

    npts = len(scan_res["tot"])
    for key in ("es", "pol", "disp", "posA", "posB"):
        if len(scan_res[key]) != npts:
            return False, f"shape mismatch: {key} has {len(scan_res[key])}, expected {npts}"
    return True, ""


def check_close(
    a: np.ndarray,
    b: np.ndarray,
    atol: float,
    rtol: float,
) -> tuple[bool, float]:
    diff = np.abs(a - b)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    return bool(np.allclose(a, b, atol=atol, rtol=rtol)), max_abs


def apply_backup_if_needed(output_path: Path, mode: str) -> Path | None:
    if not output_path.exists() or mode == "none":
        return None
    if mode == "overwrite":
        backup = output_path.with_suffix(output_path.suffix + ".bak")
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output_path.with_suffix(output_path.suffix + f".bak.{stamp}")
    shutil.copy2(output_path, backup)
    return backup


def write_report(path: Path | None, report: dict[str, Any]):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as ofile:
        json.dump(report, ofile, indent=2, sort_keys=True)


def run(cfg: Config) -> int:
    t0 = time.time()
    report: dict[str, Any] = {
        "run_meta": {
            "timestamp": dt.datetime.now().isoformat(),
            "input": str(cfg.input_path),
            "output": str(cfg.output_path),
            "ff": str(cfg.ff),
            "strict": cfg.strict,
            "fail_fast": cfg.fail_fast,
            "dry_run": cfg.dry_run,
            "backup": cfg.backup,
            "consistency_atol": cfg.consistency_atol,
            "consistency_rtol": cfg.consistency_rtol,
        },
        "selection": {},
        "stats": {
            "selected_pairs": 0,
            "processed_pairs": 0,
            "processed_batches": 0,
            "processed_points": 0,
            "warnings": 0,
            "failures": 0,
        },
        "checks": {
            "failures": [],
            "max_abs_lr_tot_consistency": 0.0,
            "max_abs_total_conservation": 0.0,
            "max_abs_component_conservation": {
                "es": 0.0,
                "pol": 0.0,
                "disp": 0.0,
            },
        },
        "artifacts": {
            "backup_file": None,
            "report_json": str(cfg.report_json) if cfg.report_json else None,
        },
    }

    try:
        data = load_pickle(cfg.input_path)
    except Exception as exc:
        report["checks"]["failures"].append(
            {"pair": "", "batch": "", "reason": "input_error", "detail": str(exc)}
        )
        report["stats"]["failures"] = len(report["checks"]["failures"])
        write_report(cfg.report_json, report)
        print(f"Input error: {exc}")
        return EXIT_INPUT_ERROR

    if not isinstance(data, dict):
        msg = f"Input root must be dict, got {type(data)}"
        add_failure(report, "", "", "input_error", msg)
        report["stats"]["failures"] = len(report["checks"]["failures"])
        write_report(cfg.report_json, report)
        print(msg)
        return EXIT_INPUT_ERROR

    selected, selection_warnings = choose_pairs(data, cfg)
    report["selection"] = {
        "pairs": selected,
        "batch": cfg.batch,
        "species_filter": cfg.species,
        "explicit_pairs": cfg.pairs,
        "warnings": selection_warnings,
    }
    report["stats"]["selected_pairs"] = len(selected)
    report["stats"]["warnings"] += len(selection_warnings)

    if not selected:
        msg = "No pairs selected. Check --pairs/--species filters."
        add_failure(report, "", "", "selection_error", msg)
        report["stats"]["failures"] = len(report["checks"]["failures"])
        write_report(cfg.report_json, report)
        print(msg)
        return EXIT_INPUT_ERROR

    try:
        params = Hamiltonian(str(cfg.ff)).getParameters()
    except Exception as exc:
        add_failure(report, "", "", "input_error", f"Failed to load FF: {exc}")
        report["stats"]["failures"] = len(report["checks"]["failures"])
        write_report(cfg.report_json, report)
        print(f"Failed to load FF: {exc}")
        return EXIT_INPUT_ERROR

    evaluators: dict[str, Any] = {}
    processed_units = 0

    for pair in selected:
        try:
            _, numb_conf, monomer_a, monomer_b = pair.split("_")
            dimer_file = cfg.dimer_bank / f"dimer_{numb_conf}_{monomer_a}_{monomer_b}.pdb"
            pdb_a_file = cfg.pdb_bank / f"{monomer_a}.pdb"
            pdb_b_file = cfg.pdb_bank / f"{monomer_b}.pdb"
            if not dimer_file.exists() or not pdb_a_file.exists() or not pdb_b_file.exists():
                missing_paths = [str(p) for p in (dimer_file, pdb_a_file, pdb_b_file) if not p.exists()]
                add_failure(
                    report,
                    pair,
                    "",
                    "input_error",
                    f"missing structure files: {missing_paths}",
                )
                if cfg.fail_fast:
                    raise RuntimeError("missing required structure files")
                continue

            base = BasePairs(str(cfg.ff), str(dimer_file), str(pdb_a_file), str(pdb_b_file))
            evaluators[pair] = jit(vmap(base.cal_e, in_axes=(None, 0, 0), out_axes=(0, 0, 0)))
        except Exception as exc:
            add_failure(report, pair, "", "runtime_error", f"failed to build evaluator: {exc}")
            if cfg.fail_fast:
                break
            continue

        batches = [cfg.batch] if cfg.batch else list(data[pair].keys())
        for batch in batches:
            if cfg.sample_limit > 0 and processed_units >= cfg.sample_limit:
                break
            if batch not in data[pair]:
                add_failure(report, pair, batch, "input_error", "batch not found")
                if cfg.fail_fast:
                    break
                continue

            scan_res = data[pair][batch]
            ok, reason = validate_scan_input(scan_res)
            if not ok:
                add_failure(report, pair, batch, "input_error", reason)
                if cfg.fail_fast:
                    break
                continue

            # Snapshot full energies before modification for conservation checks.
            full_before = np.asarray(scan_res.get("tot_full", scan_res["tot"]), dtype=np.float64)
            comp_full_before = {
                comp: np.asarray(scan_res[comp], dtype=np.float64)
                + np.asarray(scan_res.get(f"lr_{comp}", np.zeros_like(scan_res[comp])), dtype=np.float64)
                for comp in ("es", "pol", "disp")
            }

            if "tot_full" not in scan_res:
                scan_res["tot_full"] = np.asarray(scan_res["tot"], dtype=np.float64).copy()

            try:
                pos_a = jnp.asarray(scan_res["posA"])
                pos_b = jnp.asarray(scan_res["posB"])
                e_es, e_pol, e_disp = evaluators[pair](params, pos_a, pos_b)
            except Exception as exc:
                add_failure(report, pair, batch, "runtime_error", f"energy evaluation failed: {exc}")
                if cfg.fail_fast:
                    break
                continue

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

            # Strict checks
            for fld in REQUIRED_LR_FIELDS:
                if fld not in scan_res:
                    add_failure(report, pair, batch, "check_failed", f"missing field after update: {fld}")
                    if cfg.fail_fast:
                        break

            lr_ok, lr_max = check_close(
                np.asarray(scan_res["lr_tot"], dtype=np.float64),
                np.asarray(scan_res["lr_es"], dtype=np.float64)
                + np.asarray(scan_res["lr_pol"], dtype=np.float64)
                + np.asarray(scan_res["lr_disp"], dtype=np.float64),
                cfg.consistency_atol,
                cfg.consistency_rtol,
            )
            report["checks"]["max_abs_lr_tot_consistency"] = max(
                float(report["checks"]["max_abs_lr_tot_consistency"]), lr_max
            )
            if not lr_ok:
                add_failure(report, pair, batch, "check_failed", f"lr_tot consistency max_abs={lr_max:.6e}")
                if cfg.fail_fast:
                    break

            total_ok, total_max = check_close(
                np.asarray(scan_res["tot"], dtype=np.float64) + np.asarray(scan_res["lr_tot"], dtype=np.float64),
                full_before,
                cfg.consistency_atol,
                cfg.consistency_rtol,
            )
            report["checks"]["max_abs_total_conservation"] = max(
                float(report["checks"]["max_abs_total_conservation"]), total_max
            )
            if not total_ok:
                add_failure(
                    report,
                    pair,
                    batch,
                    "check_failed",
                    f"total conservation failed max_abs={total_max:.6e}",
                )
                if cfg.fail_fast:
                    break

            for comp in ("es", "pol", "disp"):
                comp_ok, comp_max = check_close(
                    np.asarray(scan_res[comp], dtype=np.float64)
                    + np.asarray(scan_res[f"lr_{comp}"], dtype=np.float64),
                    comp_full_before[comp],
                    cfg.consistency_atol,
                    cfg.consistency_rtol,
                )
                report["checks"]["max_abs_component_conservation"][comp] = max(
                    float(report["checks"]["max_abs_component_conservation"][comp]), comp_max
                )
                if not comp_ok:
                    add_failure(
                        report,
                        pair,
                        batch,
                        "check_failed",
                        f"{comp} conservation failed max_abs={comp_max:.6e}",
                    )
                    if cfg.fail_fast:
                        break

            processed_units += 1
            report["stats"]["processed_batches"] += 1
            report["stats"]["processed_points"] += len(scan_res["tot"])

        report["stats"]["processed_pairs"] += 1
        if cfg.sample_limit > 0 and processed_units >= cfg.sample_limit:
            break
        if cfg.fail_fast and report["checks"]["failures"]:
            break

    report["stats"]["failures"] = len(report["checks"]["failures"])

    backup_file: Path | None = None
    if not cfg.dry_run and report["stats"]["processed_batches"] > 0:
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        backup_file = apply_backup_if_needed(cfg.output_path, cfg.backup)
        with open(cfg.output_path, "wb") as ofile:
            pickle.dump(data, ofile)

    report["artifacts"]["backup_file"] = str(backup_file) if backup_file else None
    report["run_meta"]["elapsed_sec"] = round(time.time() - t0, 6)

    write_report(cfg.report_json, report)

    print(f"Selected pairs: {report['stats']['selected_pairs']}")
    print(
        "Processed pairs={processed_pairs}, batches={processed_batches}, points={processed_points}".format(
            **report["stats"]
        )
    )
    if cfg.dry_run:
        print("Dry-run enabled: output file not written.")
    else:
        if report["stats"]["processed_batches"] == 0:
            print("No batches processed; output file not written.")
        else:
            print(f"Saved updated data to: {cfg.output_path}")

    if cfg.report_json:
        print(f"Saved report JSON to: {cfg.report_json}")

    if report["stats"]["failures"] > 0:
        print(f"Detected {report['stats']['failures']} failure(s).")
        if cfg.strict:
            return EXIT_CHECK_FAILED

    return EXIT_OK


def main():
    cfg = parse_args()
    code = run(cfg)
    sys.exit(code)


if __name__ == "__main__":
    main()
