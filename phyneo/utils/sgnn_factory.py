from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
import pickle

import jax.numpy as jnp
import numpy as np


DEFAULT_ABN_RESIDUE_NAMES = ("PF6", "BF4", "DFP")


@dataclass(frozen=True)
class SGNNModelSpec:
    name: str
    nn: int
    max_valence: int
    params_path: str | None = None
    n_layers: tuple[int, int] = (3, 2)
    sizes: tuple[tuple[int, ...], tuple[int, ...]] = ((40, 20, 20), (20, 10))
    sigma: float = 162.13039087945623
    mu: float = 117.41975505778706
    residue_names: tuple[str, ...] = ()

    def resolved(self, params_dir: str | Path | None = None) -> "SGNNModelSpec":
        if self.params_path is None or params_dir is None:
            return self
        params_path = Path(self.params_path)
        if params_path.is_absolute():
            return self
        return replace(self, params_path=str(Path(params_dir) / params_path))


@dataclass(frozen=True)
class SGNNModelBundle:
    spec: SGNNModelSpec
    graph: object
    model: object
    params: object


@dataclass(frozen=True)
class ResidueBlock:
    name: str
    residue_index: int
    atom_start: int
    atom_stop: int

    @property
    def atom_count(self) -> int:
        return self.atom_stop - self.atom_start


SGNN_STANDARD_SPEC = SGNNModelSpec(
    name="standard",
    nn=1,
    max_valence=4,
    params_path="params_sgnn.pickle",
)

SGNN_ABN_SPEC = SGNNModelSpec(
    name="abn",
    nn=0,
    max_valence=6,
    params_path="params_sgnn_ABn.pickle",
    residue_names=DEFAULT_ABN_RESIDUE_NAMES,
)


def resolve_default_sgnn_specs(params_dir: str | Path | None = None) -> dict[str, SGNNModelSpec]:
    return {
        "standard": SGNN_STANDARD_SPEC.resolved(params_dir),
        "abn": SGNN_ABN_SPEC.resolved(params_dir),
    }


def spec_for_residue_name(
    residue_name: str,
    specs: dict[str, SGNNModelSpec] | None = None,
    params_dir: str | Path | None = None,
) -> SGNNModelSpec:
    resolved_specs = specs or resolve_default_sgnn_specs(params_dir)
    abn_spec = resolved_specs["abn"]
    if residue_name in abn_spec.residue_names:
        return abn_spec
    return resolved_specs["standard"]


def load_sgnn_params(params_path: str | Path):
    with open(params_path, "rb") as ifile:
        return pickle.load(ifile)


def build_sgnn_model_bundle(
    pdb_path: str | Path,
    spec: SGNNModelSpec,
    params_path: str | Path | None = None,
) -> SGNNModelBundle:
    from dmff.sgnn.gnn import MolGNNForce
    from dmff.sgnn.graph import from_pdb

    resolved_spec = spec.resolved(Path(params_path).parent if params_path is not None else None)
    graph = from_pdb(str(pdb_path))
    model = MolGNNForce(
        graph,
        nn=resolved_spec.nn,
        max_valence=resolved_spec.max_valence,
        n_layers=resolved_spec.n_layers,
        sizes=[tuple(layer_sizes) for layer_sizes in resolved_spec.sizes],
        sigma=resolved_spec.sigma,
        mu=resolved_spec.mu,
    )
    chosen_params_path = params_path or resolved_spec.params_path
    params = load_sgnn_params(chosen_params_path) if chosen_params_path is not None else model.params
    return SGNNModelBundle(spec=resolved_spec, graph=graph, model=model, params=params)


def find_residue_blocks(topology, residue_names: tuple[str, ...] = DEFAULT_ABN_RESIDUE_NAMES) -> list[ResidueBlock]:
    target_names = set(residue_names)
    blocks: list[ResidueBlock] = []
    for residue in topology.residues():
        if residue.name not in target_names:
            continue
        atom_indices = [atom.index for atom in residue.atoms()]
        if not atom_indices:
            continue
        expected = list(range(atom_indices[0], atom_indices[-1] + 1))
        if atom_indices != expected:
            raise ValueError(
                f"Residue {residue.name}#{residue.index} is not contiguous in atom ordering."
            )
        blocks.append(
            ResidueBlock(
                name=residue.name,
                residue_index=residue.index,
                atom_start=atom_indices[0],
                atom_stop=atom_indices[-1] + 1,
            )
        )
    return blocks


def group_residue_blocks_by_name(blocks: list[ResidueBlock]) -> dict[str, list[ResidueBlock]]:
    grouped: dict[str, list[ResidueBlock]] = defaultdict(list)
    for block in blocks:
        grouped[block.name].append(block)
    return dict(grouped)


def non_residue_atom_indices(n_atoms: int, blocks: list[ResidueBlock]) -> jnp.ndarray:
    keep_mask = np.ones(n_atoms, dtype=bool)
    for block in blocks:
        keep_mask[block.atom_start:block.atom_stop] = False
    return jnp.array(np.flatnonzero(keep_mask), dtype=jnp.int32)


def stack_positions_for_blocks(positions, blocks: list[ResidueBlock]) -> jnp.ndarray:
    if not blocks:
        raise ValueError("Cannot stack positions for an empty block list.")
    atom_count = blocks[0].atom_count
    if any(block.atom_count != atom_count for block in blocks):
        raise ValueError("All residue blocks must have the same atom count.")
    return jnp.stack([positions[block.atom_start:block.atom_stop] for block in blocks], axis=0)
