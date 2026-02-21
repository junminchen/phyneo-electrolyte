#!/usr/bin/env python
"""
Pairwise Machine Learning Potential Correction.

This module implements a neural network to learn corrections to
physics-based pairwise potentials.
"""
import os
import numpy as np
import pandas as pd
import time
import scipy
import sys
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from IPython.display import clear_output

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap, random, config

from flax import linen as nn
from flax.training import train_state
import optax

from dmff.api import Hamiltonian
from dmff.utils import jit_condition, regularize_pairs, pair_buffer_scales
from dmff.admp.pairwise import distribute_scalar, distribute_v3
from dmff.admp.spatial import pbc_shift
from dmff.common import nblist

from openmm import *
from openmm.app import *
from openmm.unit import *

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Any
from scipy.sparse import csr_matrix
from functools import partial

# Import utility functions
from phyneo.utils.data_utils import (
    torch_batch_to_jax, setup_plot_style, plot_training_progress,
    get_topology_neighbors, filter_and_pad_pairs, cutoff_cosine,
    int_to_onehot, get_data, parameter_shapes, zindex, charge_to_index,
)

@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, None, None), out_axes=(0, 0, 0, 0))
def get_environment_atoms(pairs, topo_nblist, topo_mask):
    j_centers = pairs[0]
    k_centers = pairs[1]
    
    j_neighbors = jnp.take(topo_nblist, j_centers, axis=0)
    k_neighbors = jnp.take(topo_nblist, k_centers, axis=0)

    valid_j = j_neighbors != -1
    valid_k = k_neighbors != -1

    mask_j = (j_neighbors != j_centers) & (j_neighbors != k_centers) & valid_j
    mask_k = (k_neighbors != j_centers) & (k_neighbors != k_centers) & valid_k

    topo_mask_j = jnp.take(topo_mask, j_centers, axis=0)
    topo_mask_k = jnp.take(topo_mask, k_centers, axis=0)

    valid_mask_j = topo_mask_j & mask_j
    valid_mask_k = topo_mask_k & mask_k

    return j_neighbors, k_neighbors, valid_mask_j, valid_mask_k

class EAPNNForce(nn.Module):
    n_atype: int
    rc: float
    n_atoms: int
    acsf_nmu: int
    apsf_nmu: int
    acsf_eta: float
    apsf_eta: float
    use_pbc: bool = True  # New parameter, use periodic boundary conditions by default

    def setup(self):
        self.feature_extractor = FeatureExtractor(
            n_atoms=self.n_atoms,
            n_atype=self.n_atype, 
            rc=self.rc, 
            acsf_nmu=self.acsf_nmu,
            apsf_nmu=self.apsf_nmu,
            acsf_eta=self.acsf_eta,
            apsf_eta=self.apsf_eta,
            use_pbc=self.use_pbc  # New parameter, use periodic boundary conditions by default
        )
        self.neural_network = NeuralNetwork()

    def __call__(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        features, dr_norm, buffer_scales = self.feature_extractor(pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
        atomic_energies = self.neural_network(features, dr_norm, buffer_scales)
        return jnp.sum(atomic_energies)

    def get_features(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        return self.feature_extractor(pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)

    def get_energy(self, features, dr_norm, buffer_scales):
        atomic_energies = self.neural_network(features, dr_norm, buffer_scales)
        return jnp.sum(atomic_energies)

    def predict_energy_force(self, params, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        """Simultaneously predict energy and force (force = -dE/dpos)"""
        def energy_fn(pos):
            return self.apply(
                params, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices
            )
        
        energy, grad_energy = value_and_grad(energy_fn)(pos)
        force = grad_energy  # Physical definition of force
        return energy, force

class FeatureExtractor(nn.Module):
    n_atoms: int
    n_atype: int
    rc: float
    acsf_nmu: int = 20
    apsf_nmu: int = 10
    acsf_eta: float = 100
    apsf_eta: float = 25
    use_pbc: bool = True  # New parameter, use periodic boundary conditions by default

    def setup(self):
        self.acsf_mus = jnp.linspace(0.0, 5.0, self.acsf_nmu)
        self.apsf_mus = jnp.linspace(-1.0, 1.0, self.apsf_nmu)

    def compute_atomcenter_features(self, pos, box, topo_nblist, topo_mask, atype_indices, acsf_mus, acsf_eta):
        """Directly calculate environmental features for all atoms"""
        # Get environmental atom positions [n_atoms, max_neighbors, 3]
        r_center = pos  # [n_atoms, 3]
        r_env = pos[topo_nblist]  # [n_atoms, max_neighbors, 3]
        
        # Calculate relative positions and distances
        dr = r_env - r_center[:, None, :]  # [n_atoms, max_neighbors, 3]
        box_inv = jnp.linalg.inv(box)
        dr = pbc_shift(dr, box, box_inv)  # If using PBC, box and box_inv need to be passed here
        dr_norm = jnp.linalg.norm(dr+1e-10, axis=2)  # [n_atoms, max_neighbors]
        
        # Calculate cutoff function
        f_cut = cutoff_cosine(dr_norm, self.rc) * topo_mask  # [n_atoms, max_neighbors]
        
        # Calculate radial basis functions
        exp_term = jnp.exp(-acsf_eta * jnp.square(dr_norm[..., None] - acsf_mus))  # [n_atoms, max_neighbors, n_mu]
        G_raw = exp_term * f_cut[..., None]  # [n_atoms, max_neighbors, n_mu]
        
        # Accumulate features by atom type
        type_one_hot = (atype_indices[topo_nblist][..., None] == jnp.arange(self.n_atype))  # [n_atoms, max_neighbors, n_atype]
        
        # Calculate all features at once
        G = jnp.einsum('ijk,ijl->ikl', G_raw, type_one_hot)  # [n_atoms, n_mu, n_atype]
        
        return G
    
    def compute_atompair_features(self, cos_gamma_i, cos_gamma_j, j_list, k_list, j_mask, k_mask,
                                  buffer_nblist_inter_rc, atype_indices, apsf_mus, apsf_eta):
        # Calculate angle features for i and j
        angle_features_i = jnp.exp(-apsf_eta * jnp.square(cos_gamma_i[..., None] - apsf_mus))
        angle_features_j = jnp.exp(-apsf_eta * jnp.square(cos_gamma_j[..., None] - apsf_mus))

        # Create type_one_hot [n_pairs, max_neighbors, n_atype]
        type_one_hot_i = (atype_indices[j_list][..., None] == jnp.arange(self.n_atype))
        type_one_hot_j = (atype_indices[k_list][..., None] == jnp.arange(self.n_atype))

        # Apply mask
        masked_features_i = angle_features_i * j_mask[..., None]  # [n_pairs, max_neighbors, n_mu]
        masked_features_j = angle_features_j * k_mask[..., None]  # [n_pairs, max_neighbors, n_mu]

        # Calculate contributions of all types at once
        G_i = jnp.einsum('ijk,ijl->ikl', masked_features_i, type_one_hot_i)  # [n_pairs, n_mu, n_atype]
        G_j = jnp.einsum('ijk,ijl->ikl', masked_features_j, type_one_hot_j)  # [n_pairs, n_mu, n_atype]

        # Symmetric average and apply intermolecular interaction mask
        G = (G_i + G_j) * 0.5 * buffer_nblist_inter_rc[:, None, None]

        return G

    def __call__(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        apsf_mus, apsf_eta = self.apsf_mus, self.apsf_eta
        acsf_mus, acsf_eta = self.acsf_mus, self.acsf_eta
        
        mScales = jnp.array([0., 0., 0., 0., 0., 1.])
        pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
        nbonds = pairs[:, 2]
        mscales = distribute_scalar(mScales, nbonds - 1)
        pairs = pairs[:, :2]
        buffer_scales = pair_buffer_scales(pairs[:, :2]) * valid_mask
        mscales = mscales * buffer_scales

        box_inv = jnp.linalg.inv(box)
        ri = pos[pairs[:, 0]]
        rj = pos[pairs[:, 1]]

        rij = rj - ri
        rij = pbc_shift(rij, box, box_inv)

        dr_norm = jnp.linalg.norm(rij + 1e-10, axis=1)

        same_mol = mol_ID[pairs[:, 0]] == mol_ID[pairs[:, 1]]
        buffer_inter = jnp.where(same_mol, 0., 1.)
        buffer_intra = jnp.where(same_mol, 1., 0.)
        cutoff = 0.5 * (1 + jnp.cos(jnp.pi * dr_norm / self.rc))
        cutoff = jnp.where(dr_norm <= self.rc, cutoff, 0.0)

        buffer_nblist_inter = buffer_inter * buffer_scales
        buffer_nblist_intra = buffer_intra * buffer_scales
        buffer_nblist_inter_rc = buffer_nblist_inter * cutoff


        # Get environment atom information
        j_list, k_list, j_mask, k_mask = get_environment_atoms(pairs, topo_nblist, topo_mask)

        # Calculate positions and angles of environment atoms (both directions)
        # i environment
        valid_j_mask = j_mask[..., None]  # (n_pairs, max_neighbors, 1)
        rj_env = jnp.where(valid_j_mask, pos[j_list], 0.0)  # Set invalid index positions to 0
        # rj_env = pos[j_list]
        rj_X = rj_env - ri[:, None, :]
        rj_X = pbc_shift(rj_X, box, box_inv)
        norm_rj_X = jnp.linalg.norm(rj_X + 1e-10, axis=2, keepdims=True)
        rj_X_norm = rj_X / norm_rj_X
        rij_unit = rij / (dr_norm[:, None] + 1e-10)
        cos_gamma_i = jnp.einsum('aji,ai->aj', rj_X_norm, rij_unit) * j_mask

        # j environment
        valid_k_mask = k_mask[..., None]  # (n_pairs, max_neighbors, 1)
        rk_env = jnp.where(valid_k_mask, pos[k_list], 0.0)  # Set invalid index positions to 0
        # rk_env = pos[k_list]
        rk_X = rk_env - rj[:, None, :]
        rk_X = pbc_shift(rk_X, box, box_inv)
        norm_rk_X = jnp.linalg.norm(rk_X + 1e-10, axis=2, keepdims=True)
        rk_X_norm = rk_X / norm_rk_X
        rji_unit = -rij_unit
        cos_gamma_j = jnp.einsum('aji,ai->aj', rk_X_norm, rji_unit) * k_mask

        # Calculate atom pair features
        atompair_features = self.compute_atompair_features(cos_gamma_i, cos_gamma_j, j_list, k_list, j_mask, k_mask,
                                                           buffer_nblist_inter_rc, atype_indices, apsf_mus, apsf_eta)

        atom_features = self.compute_atomcenter_features(
            pos, box, topo_nblist, topo_mask, atype_indices, acsf_mus, acsf_eta)
        
        atom_features_i = atom_features[pairs[:, 0],]
        atom_features_j = atom_features[pairs[:, 1],]
        atom_features = (atom_features_i + atom_features_j) * 0.5

        # Process atom type features
        elem_indices = jnp.array(zindex)[atype_indices]
        j_atype = elem_indices[pairs[:,0]]  # type of atom j
        k_atype = elem_indices[pairs[:,1]]  # type of atom k
        
        # Create one-hot encoding for atoms j and k separately
        j_onehot = jnp.concatenate([j_atype.reshape(-1,1), int_to_onehot(j_atype, 10)], axis=1)
        k_onehot = jnp.concatenate([k_atype.reshape(-1,1), int_to_onehot(k_atype, 10)], axis=1)
        
        # Merge type features of j and k
        atype_onehot = jnp.concatenate([j_onehot, k_onehot], axis=1)

        atom_features = atom_features.reshape(atom_features.shape[0], -1)
        atompair_features = atompair_features.reshape(atompair_features.shape[0], -1)
        apsf_features = jnp.concatenate((atom_features, atompair_features, atype_onehot), axis=1)

        return apsf_features, dr_norm, buffer_nblist_inter_rc    

class NeuralNetwork(nn.Module):
    dense_nodes: int = 64
    
    @nn.compact
    def __call__(self, combined, dr_norm, buffer_nblist_inter):
        x = combined
        for _ in range(3):
            x = nn.Dense(self.dense_nodes)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        out_AB = nn.Dense(1)(x)
        
        return jnp.sum(out_AB * buffer_nblist_inter[:,None]) 