#!/usr/bin/env python
"""Unified CLI entrypoint for example workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_script(script: Path, script_args: list[str], dry_run: bool) -> int:
    cmd = [sys.executable, str(script), *script_args]
    print(f"[examples-cli] cwd={script.parent}")
    print(f"[examples-cli] run={' '.join(cmd)}")
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=script.parent, check=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CLI for phyneo-electrolyte examples"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target command without executing",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_cmd(name: str, help_text: str, target: str) -> None:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument(
            "script_args",
            nargs=argparse.REMAINDER,
            help="Arguments passed through to the target script",
        )
        sub.set_defaults(target_script=target)

    add_cmd(
        "slater-train",
        "Run Slater dimer training",
        "1_training_slater_nb/train_dimer.py",
    )
    add_cmd(
        "slater-gen-xml",
        "Generate XML from fitted Slater params",
        "1_training_slater_nb/gen_xml.py",
    )
    add_cmd(
        "pairwise-train",
        "Run pairwise EAPNN training",
        "2_training_pairwise_ml_nb/train_eapnn.py",
    )
    add_cmd(
        "sgnn-train",
        "Run sGNN bonding training",
        "3_training_sgnn_bonding/train_total.py",
    )
    add_cmd(
        "sgnn-abn-train",
        "Run ABn sGNN bonding training",
        "3_training_sgnn_bonding/train_abn.py",
    )
    add_cmd(
        "sgnn-joint-train",
        "Run joint standard + ABn sGNN training",
        "3_training_sgnn_bonding/train_joint.py",
    )
    add_cmd(
        "sgnn-plot",
        "Plot sGNN training output",
        "3_training_sgnn_bonding/plot_data.py",
    )
    add_cmd(
        "md-gen-pdb",
        "Generate MD initial PDB",
        "md_simulation/gen_md_pdb.py",
    )
    add_cmd(
        "md-driver",
        "Run MD simulation driver",
        "md_simulation/driver.py",
    )
    add_cmd(
        "md-client",
        "Run MD client with DMFF",
        "md_simulation/client_dmff.py",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    examples_dir = Path(__file__).resolve().parent
    target_script = examples_dir / args.target_script
    if not target_script.exists():
        parser.error(f"target script not found: {target_script}")

    passthrough = args.script_args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    return _run_script(target_script, passthrough, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
