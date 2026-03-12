import os

os.environ["JAX_PLATFORMS"] = "cpu"

import sys
from pathlib import Path

import numpy as np
from openmm.app import PDBFile

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_DIR))

from phyneo.utils import (
    DEFAULT_ABN_RESIDUE_NAMES,
    build_sgnn_model_bundle,
    find_residue_blocks,
    group_residue_blocks_by_name,
    resolve_default_sgnn_specs,
    spec_for_residue_name,
)


def test_spec_resolution():
    specs = resolve_default_sgnn_specs(REPO_DIR / "examples" / "md_simulation")

    standard = specs["standard"]
    abn = spec_for_residue_name("PF6", specs)

    assert standard.nn == 1
    assert standard.max_valence == 4
    assert abn.nn == 0
    assert abn.max_valence == 6
    assert Path(abn.params_path).name == "params_sgnn_ABn.pickle"


def test_build_sgnn_model_bundle():
    specs = resolve_default_sgnn_specs(REPO_DIR / "examples" / "md_simulation")

    ec_bundle = build_sgnn_model_bundle(REPO_DIR / "data" / "pdb_bank" / "EC.pdb", specs["standard"])
    pf6_bundle = build_sgnn_model_bundle(
        REPO_DIR / "examples" / "md_simulation" / "pdb_bank" / "PF6.pdb",
        spec_for_residue_name("PF6", specs),
    )

    assert ec_bundle.model.nn == 1
    assert ec_bundle.graph.max_valence == 4
    assert pf6_bundle.model.nn == 0
    assert pf6_bundle.graph.max_valence == 6


def test_find_abn_residue_blocks():
    pdb = PDBFile(str(REPO_DIR / "examples" / "md_simulation" / "pdb_bank" / "PF6.pdb"))
    blocks = find_residue_blocks(pdb.topology, DEFAULT_ABN_RESIDUE_NAMES)
    grouped = group_residue_blocks_by_name(blocks)

    assert len(blocks) == 1
    assert blocks[0].name == "PF6"
    assert blocks[0].atom_count == len(list(pdb.topology.atoms()))
    assert list(grouped.keys()) == ["PF6"]


if __name__ == "__main__":
    test_spec_resolution()
    test_build_sgnn_model_bundle()
    test_find_abn_residue_blocks()
    print("All sGNN factory tests passed.")
