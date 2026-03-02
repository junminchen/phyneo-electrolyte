#!/usr/bin/env python
"""Generate updated FF XML from backend-trained FF-tree parameter pickle."""

from __future__ import annotations

import argparse
import os
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from dmff.api import Hamiltonian

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


def load_ff_params_tree(ff_xml: str, params_pickle: str):
    base_tree = tree_to_jax(Hamiltonian(ff_xml).getParameters().parameters)
    with open(params_pickle, "rb") as ifile:
        loaded = pickle.load(ifile)

    if hasattr(loaded, "parameters"):
        return tree_to_jax(loaded.parameters)
    if isinstance(loaded, dict):
        if all(k in loaded and isinstance(loaded[k], dict) for k in FORCES_WITH_A):
            return tree_to_jax(loaded)
        return apply_legacy_restart(base_tree, loaded)
    raise ValueError(f"Unsupported params pickle format: {type(loaded)}")


def update_xml_with_tree(input_xml: str, params_tree: dict[str, dict[str, jnp.ndarray]], output_xml: str):
    tree = ET.parse(input_xml)
    root = tree.getroot()

    updated_stats: dict[str, int] = {}
    for force_name, force_params in params_tree.items():
        for force_elem in root.iter(force_name):
            atoms = list(force_elem.iter("Atom"))
            for param_name, arr in force_params.items():
                arr = jnp.asarray(arr)
                if arr.ndim != 1:
                    continue
                target_atoms = [atom for atom in atoms if param_name in atom.attrib]
                n_targets = len(target_atoms)
                if n_targets == 0:
                    continue
                if arr.shape[0] < n_targets:
                    raise ValueError(
                        f"{force_name}.{param_name} has length {arr.shape[0]}, fewer than XML target atoms {n_targets}"
                    )
                for idx, atom_elem in enumerate(target_atoms):
                    atom_elem.set(param_name, str(float(arr[idx])))
                    updated_stats[f"{force_name}.{param_name}"] = (
                        updated_stats.get(f"{force_name}.{param_name}", 0) + 1
                    )

    out_path = Path(output_xml)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_path))

    print(f"Saved updated force field XML to: {out_path}")
    print("Updated fields:")
    for k in sorted(updated_stats):
        print(f"  {k}: {updated_stats[k]}")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate updated XML from FF-tree params pickle")
    parser.add_argument("--params", default=str(script_dir / "params" / "params_ff_backend.pickle"))
    parser.add_argument("--input-xml", default=str(script_dir / "phyneo_ecl.xml"))
    parser.add_argument("--output-xml", default=str(script_dir / "output" / "output_ff_backend.xml"))
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(Path(args.output_xml).parent, exist_ok=True)
    params_tree = load_ff_params_tree(args.input_xml, args.params)
    update_xml_with_tree(args.input_xml, params_tree, args.output_xml)

    # Validate XML can be loaded by DMFF.
    _ = Hamiltonian(args.output_xml)
    print("Validation passed: Hamiltonian can load generated XML.")


if __name__ == "__main__":
    main()
