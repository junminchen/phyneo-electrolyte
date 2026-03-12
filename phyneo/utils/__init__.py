"""
Utility functions for PhyNEO-Electrolyte.
"""

from .data_utils import (
    # New exports
    MoleculeTorchDataset,
    torch_batch_to_jax,
    get_topology_neighbors,
    filter_and_pad_pairs,
    int_to_onehot,
    get_data,
    parameter_shapes,
    zindex,
    charge_to_index,
    # Plotting
    setup_plot_style,
    plot_training_progress
)
from .sgnn_factory import (
    DEFAULT_ABN_RESIDUE_NAMES,
    SGNN_ABN_SPEC,
    SGNN_STANDARD_SPEC,
    SGNNModelBundle,
    SGNNModelSpec,
    ResidueBlock,
    build_sgnn_model_bundle,
    find_residue_blocks,
    group_residue_blocks_by_name,
    load_sgnn_params,
    non_residue_atom_indices,
    resolve_default_sgnn_specs,
    spec_for_residue_name,
    stack_positions_for_blocks,
)

__all__ = [
    "MoleculeTorchDataset",
    "torch_batch_to_jax",
    "get_topology_neighbors",
    "filter_and_pad_pairs",
    "int_to_onehot",
    "get_data",
    "parameter_shapes",
    "zindex",
    "charge_to_index",
    "setup_plot_style",
    "plot_training_progress",
    "DEFAULT_ABN_RESIDUE_NAMES",
    "SGNN_ABN_SPEC",
    "SGNN_STANDARD_SPEC",
    "SGNNModelBundle",
    "SGNNModelSpec",
    "ResidueBlock",
    "build_sgnn_model_bundle",
    "find_residue_blocks",
    "group_residue_blocks_by_name",
    "load_sgnn_params",
    "non_residue_atom_indices",
    "resolve_default_sgnn_specs",
    "spec_for_residue_name",
    "stack_positions_for_blocks",
]
