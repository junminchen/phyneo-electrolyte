import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import MDAnalysis as mda
from scipy.sparse import csr_matrix
from typing import Dict, Any, List

# Constants
zindex = [1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 15.0, 16.0]
charge_to_index = {
    0.0: 100000, 1.0: 0, 3.0: 1, 5.0: 2, 6.0: 3,
    7.0: 4, 8.0: 5, 9.0: 6, 11.0: 7, 15.0: 8, 16.0: 9
}

def torch_batch_to_jax(batch):
    jax_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            jax_batch[k] = jnp.array(v.numpy())
        else:
            jax_batch[k] = jnp.array(v)
    return jax_batch

def setup_plot_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'mathtext.fontset': 'custom',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.max_open_warning': 50,
        'figure.dpi': 100
    })
    return fm.FontProperties(family='DejaVu Sans', size=12)

def plot_training_progress(epoch, num_epochs, train_metrics, test_metrics, total_time):
    english_font = setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    epochs = list(range(0, len(train_metrics['total_loss'])*10, 10))
    ax1.plot(epochs, train_metrics['total_loss'], 'r-', marker='o', label='Total Loss', markersize=4)
    ax1.plot(epochs, train_metrics['energy_loss'], 'b--', label='Energy Loss', linewidth=2)
    if any(m > 0 for m in train_metrics['force_loss']):
        ax1.plot(epochs, train_metrics['force_loss'], 'g-.', label='Force Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontproperties=english_font)
    ax1.set_ylabel('Loss', fontproperties=english_font)
    ax1.set_title(f'Training Loss (Total Time: {total_time/60:.1f} min)', fontproperties=english_font)
    ax1.legend(prop=english_font)
    ax1.grid(alpha=0.3)
    
    # Test set RMSE
    ax2_twin = ax2.twinx()
    ax2.plot(epochs, test_metrics['energy_rmse'], 'b-', marker='s', label='Energy RMSE', markersize=4)
    if any(m > 0 for m in test_metrics['force_rmse']):
        ax2_twin.plot(epochs, test_metrics['force_rmse'], 'g-', marker='^', label='Force RMSE', markersize=4)
        ax2_twin.set_ylabel('Force RMSE (kJ/(mol·Å))', color='g', fontproperties=english_font)
        ax2_twin.tick_params(axis='y', labelcolor='g')
    
    ax2.set_xlabel('Epoch', fontproperties=english_font)
    ax2.set_ylabel('Energy RMSE (kJ/mol)', color='b', fontproperties=english_font)
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_title('Test Set RMSE', fontproperties=english_font)
    ax2.grid(alpha=0.3)
    
    # Merge legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', prop=english_font)
    
    plt.tight_layout()
    return fig

def get_topology_neighbors(pdb_file, connectivity=2, max_neighbors=18, max_n_atoms=None):
    mol = mda.Universe(pdb_file)
    n_atoms = len(mol.atoms)
    if max_n_atoms is None:
        max_n_atoms = n_atoms

    indices = np.full((max_n_atoms, max_neighbors), -1, dtype=np.int32)
    mask = np.zeros((max_n_atoms, max_neighbors), dtype=np.int32)

    try:
        has_bonds = len(mol.bonds) > 0
    except AttributeError:
        has_bonds = False

    if has_bonds:
        row_idx, col_idx = [], []
        for bond in mol.bonds:
            i, j = bond.atoms[0].index, bond.atoms[1].index
            row_idx.extend([i, j])
            col_idx.extend([j, i])
        data = np.ones(len(row_idx), dtype=bool)
        adj_init = csr_matrix((data, (row_idx, col_idx)), shape=(n_atoms, n_atoms), dtype=bool)

        adj_matrix_odd = adj_init.copy()
        adj_matrix_self_even = adj_init.copy()
        adj_matrix = adj_matrix_odd.copy()

        for _ in range(connectivity - 1):
            adj_matrix_self_even = (adj_matrix_self_even @ adj_matrix_self_even).astype(bool)
            adj_matrix = (adj_matrix_odd + adj_matrix_self_even).astype(bool)
            adj_matrix_odd = (adj_matrix_self_even @ adj_init).astype(bool)

        for i in range(n_atoms):
            neighbors = adj_matrix[i].indices
            neighbors = neighbors[neighbors != i]
            n_real = min(len(neighbors), max_neighbors)
            indices[i, :n_real] = neighbors[:n_real]
            mask[i, :n_real] = 1

    return indices, mask

def filter_and_pad_pairs(pairs, atype_indices, target_atype_indices, max_pairs=30):
    pair_atype_i = atype_indices[pairs[:, 0]]
    pair_atype_j = atype_indices[pairs[:, 1]]
    is_target_i = jnp.isin(pair_atype_i, target_atype_indices)
    is_target_j = jnp.isin(pair_atype_j, target_atype_indices)
    target_pair_mask = is_target_i | is_target_j
    filtered_pairs = pairs[target_pair_mask]
    n_target_pairs = filtered_pairs.shape[0]

    if n_target_pairs < max_pairs:
        pad_pairs = jnp.full((max_pairs - n_target_pairs, 3), -1, dtype=pairs.dtype)
        padded_pairs = jnp.concatenate([filtered_pairs, pad_pairs], axis=0)
        valid_mask_int = jnp.concatenate([
            jnp.ones(n_target_pairs, dtype=jnp.int32),
            jnp.zeros(max_pairs - n_target_pairs, dtype=jnp.int32)
        ])
    else:
        padded_pairs = filtered_pairs
        valid_mask_int = jnp.ones(len(filtered_pairs), dtype=jnp.int32)
        
    return padded_pairs, valid_mask_int

def cutoff_cosine(distances, cutoff):
    x = distances / cutoff
    return jnp.where(x < 1, 0.5 * (jnp.cos(jnp.pi * x) + 1), 0.0)

def int_to_onehot(labels, charge_to_index, num_classes: int):
    charges = jnp.array(list(charge_to_index.keys()))
    indices = jnp.array(list(charge_to_index.values()))
    z_indices = jnp.take(indices, jnp.searchsorted(charges, labels))
    return jax.nn.one_hot(z_indices, num_classes)

def get_data(data, arr):
    dimer_test = [key for key in data if key.split('_')[-2] in arr and key.split('_')[-1] in arr]
    return dimer_test

def parameter_shapes(params):
    return jax.tree_util.tree_map(lambda p: p.shape, params)
