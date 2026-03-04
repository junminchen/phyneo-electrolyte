#!/usr/bin/env python
"""
Environment-Aware Pairwise Neural Network (EAPNN) for Potential Correction.

This module implements both the optimized neural network architecture (EAPNNForce)
and the legacy architecture (EAPNNForceLegacy) for backward compatibility with 
the OpenMM-ML interface.
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
from typing import Tuple, Dict, List, Any, Optional
from functools import partial

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
from scipy.sparse import csr_matrix

# Import utility functions
from phyneo.utils.data_utils import (
    torch_batch_to_jax, setup_plot_style, plot_training_progress,
    get_topology_neighbors, filter_and_pad_pairs, cutoff_cosine,
    int_to_onehot, get_data, parameter_shapes, zindex, charge_to_index
)

@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, None, None), out_axes=(0, 0, 0, 0))
def get_environment_atoms(pairs, topo_nblist, topo_mask):
    """
    Extract neighbor indices and masks for a set of pairs.
    """
    j_centers = pairs[0]
    k_centers = pairs[1]
    
    j_neighbors = jnp.take(topo_nblist, j_centers, axis=0)
    k_neighbors = jnp.take(topo_nblist, k_centers, axis=0)

    valid_j = j_neighbors != -1
    valid_k = k_neighbors != -1

    # Exclude the pair partners from each other's environment to avoid self-interaction in features
    mask_j = (j_neighbors != j_centers) & (j_neighbors != k_centers) & valid_j
    mask_k = (k_neighbors != j_centers) & (k_neighbors != k_centers) & valid_k

    topo_mask_j = jnp.take(topo_mask, j_centers, axis=0)
    topo_mask_k = jnp.take(topo_mask, k_centers, axis=0)

    valid_mask_j = topo_mask_j & mask_j
    valid_mask_k = topo_mask_k & mask_k

    return j_neighbors, k_neighbors, valid_mask_j, valid_mask_k

# =============================================================================
# OPTIMIZED ARCHITECTURE (from feat/eapnn-optimized-architecture)
# =============================================================================

class BesselBasis(nn.Module):
    n_basis: int
    rc: float
    def setup(self):
        self.frequencies = jnp.pi * jnp.arange(1, self.n_basis + 1) / self.rc
    def __call__(self, x):
        return jnp.sqrt(2 / self.rc) * jnp.sin(self.frequencies * x) / (x + 1e-10)

class AngularBasis(nn.Module):
    n_basis: int
    def setup(self):
        self.mus = jnp.linspace(-1.0, 1.0, self.n_basis)
        self.eta = 2.0 / (self.mus[1] - self.mus[0])**2
    def __call__(self, cos_theta):
        return jnp.exp(-self.eta * jnp.square(cos_theta - self.mus))

class ResidualBlock(nn.Module):
    features: int
    dropout_rate: float = 0.0
    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        x = nn.Dense(self.features)(x)
        x = nn.LayerNorm()(x)
        x = nn.swish(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.features)(x)
        return x + residual

class AttentionInteraction(nn.Module):
    embed_dim: int
    num_heads: int = 4
    @nn.compact
    def __call__(self, query, key_value, mask):
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, deterministic=True, dtype=jnp.float32
        )(query, key_value, mask=mask[:, None, None, :])
        return jnp.squeeze(attn_out, axis=1)

class EAPNNForce(nn.Module):
    n_atype: int
    rc: float
    n_atoms: int
    embed_dim: int = 32
    n_radial: int = 16
    n_angular: int = 8
    n_layers: int = 3
    hidden_dim: int = 128
    use_pbc: bool = True

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.n_atype + 1, features=self.embed_dim)
        self.radial_basis = BesselBasis(n_basis=self.n_radial, rc=self.rc)
        self.angular_basis = AngularBasis(n_basis=self.n_angular)
        self.attention = AttentionInteraction(embed_dim=self.embed_dim)
        self.dense_v_env = nn.Dense(self.embed_dim)
        self.dense_q = nn.Dense(self.embed_dim)
        self.dense_combined = nn.Dense(self.hidden_dim)
        layers = [ResidualBlock(features=self.hidden_dim) for _ in range(self.n_layers)]
        self.mlp = nn.Sequential(layers)
        self.final_dense = nn.Dense(1)

    def _compute_internal(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices, train: bool = False):
        box_inv = jnp.linalg.inv(box)
        pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
        buffer_scales = pair_buffer_scales(pairs[:, :2]) * valid_mask
        ri, rj = pos[pairs[:, 0]], pos[pairs[:, 1]]
        rij = pbc_shift(rj - ri, box, box_inv)
        dr_norm = jnp.linalg.norm(rij + 1e-10, axis=1, keepdims=True)
        rij_unit = rij / dr_norm
        same_mol = mol_ID[pairs[:, 0]] == mol_ID[pairs[:, 1]]
        buffer_inter = jnp.where(same_mol, 0., 1.)
        f_cut = cutoff_cosine(dr_norm, self.rc)
        mask_final = buffer_inter * buffer_scales[:, None] * f_cut
        safe_atypes = jnp.where(atype_indices == 10000, self.n_atype, atype_indices)
        h = self.embedding(safe_atypes)
        hi, hj = h[pairs[:, 0]], h[pairs[:, 1]]
        j_list, k_list, j_mask, k_mask = get_environment_atoms(pairs, topo_nblist, topo_mask)
        ri_env = pos[j_list]
        ri_X = pbc_shift(ri_env - ri[:, None, :], box, box_inv)
        di_X_norm = jnp.linalg.norm(ri_X + 1e-10, axis=2, keepdims=True)
        cos_gamma_i = jnp.einsum('ajk,ak->aj', ri_X / di_X_norm, rij_unit)
        feat_i_env = jnp.concatenate([self.radial_basis(di_X_norm), self.angular_basis(cos_gamma_i[..., None])], axis=-1)
        v_env_i = self.dense_v_env(feat_i_env) * h[j_list]
        env_agg_i = self.attention(self.dense_q(hi[:, None, :]), v_env_i, j_mask)
        rj_env = pos[k_list]
        rj_X = pbc_shift(rj_env - rj[:, None, :], box, box_inv)
        dj_X_norm = jnp.linalg.norm(rj_X + 1e-10, axis=2, keepdims=True)
        cos_gamma_j = jnp.einsum('ajk,ak->aj', rj_X / dj_X_norm, -rij_unit)
        feat_j_env = jnp.concatenate([self.radial_basis(dj_X_norm), self.angular_basis(cos_gamma_j[..., None])], axis=-1)
        v_env_j = self.dense_v_env(feat_j_env) * h[k_list]
        env_agg_j = self.attention(self.dense_q(hj[:, None, :]), v_env_j, k_mask)
        combined = jnp.concatenate([hi, hj, self.radial_basis(dr_norm), env_agg_i, env_agg_j], axis=-1)
        return combined, dr_norm, mask_final

    def __call__(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices, train: bool = False):
        combined, _, mask_final = self._compute_internal(pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices, train=train)
        x = self.dense_combined(combined)
        x = self.mlp(x, train=train)
        return jnp.sum(self.final_dense(x) * mask_final)

    def predict_energy_force(self, params, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        def energy_fn(p): return self.apply(params, p, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
        energy, grad_energy = value_and_grad(energy_fn)(pos)
        return energy, -grad_energy

# =============================================================================
# LEGACY ARCHITECTURE (from feature/openmm-ml-interface)
# =============================================================================

class FeatureExtractor(nn.Module):
    n_atoms: int
    n_atype: int
    rc: float
    acsf_nmu: int = 20
    apsf_nmu: int = 10
    acsf_eta: float = 100
    apsf_eta: float = 25
    use_pbc: bool = True

    def setup(self):
        self.acsf_mus = jnp.linspace(0.0, 5.0, self.acsf_nmu)
        self.apsf_mus = jnp.linspace(-1.0, 1.0, self.apsf_nmu)

    def compute_atomcenter_features(self, pos, box, topo_nblist, topo_mask, atype_indices, acsf_mus, acsf_eta):
        r_center = pos
        r_env = pos[topo_nblist]
        dr = pbc_shift(r_env - r_center[:, None, :], box, jnp.linalg.inv(box))
        dr_norm = jnp.linalg.norm(dr+1e-10, axis=2)
        f_cut = cutoff_cosine(dr_norm, self.rc) * topo_mask
        G_raw = jnp.exp(-acsf_eta * jnp.square(dr_norm[..., None] - acsf_mus)) * f_cut[..., None]
        type_one_hot = (atype_indices[topo_nblist][..., None] == jnp.arange(self.n_atype))
        return jnp.einsum('ijk,ijl->ikl', G_raw, type_one_hot)
    
    def compute_atompair_features(self, cos_gamma_i, cos_gamma_j, j_list, k_list, j_mask, k_mask,
                                  buffer_nblist_inter_rc, atype_indices, apsf_mus, apsf_eta):
        feat_i = jnp.exp(-apsf_eta * jnp.square(cos_gamma_i[..., None] - apsf_mus)) * j_mask[..., None]
        feat_j = jnp.exp(-apsf_eta * jnp.square(cos_gamma_j[..., None] - apsf_mus)) * k_mask[..., None]
        G_i = jnp.einsum('ijk,ijl->ikl', feat_i, (atype_indices[j_list][..., None] == jnp.arange(self.n_atype)))
        G_j = jnp.einsum('ijk,ijl->ikl', feat_j, (atype_indices[k_list][..., None] == jnp.arange(self.n_atype)))
        return (G_i + G_j) * 0.5 * buffer_nblist_inter_rc[:, None, None]

    def __call__(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        box_inv = jnp.linalg.inv(box)
        pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
        buffer_scales = pair_buffer_scales(pairs[:, :2]) * valid_mask
        ri, rj = pos[pairs[:, 0]], pos[pairs[:, 1]]
        rij = pbc_shift(rj - ri, box, box_inv)
        dr_norm = jnp.linalg.norm(rij + 1e-10, axis=1)
        same_mol = mol_ID[pairs[:, 0]] == mol_ID[pairs[:, 1]]
        buffer_nblist_inter_rc = jnp.where(same_mol, 0., 1.) * buffer_scales * jnp.where(dr_norm <= self.rc, 0.5*(1+jnp.cos(jnp.pi*dr_norm/self.rc)), 0.0)
        j_list, k_list, j_mask, k_mask = get_environment_atoms(pairs, topo_nblist, topo_mask)
        rij_unit = rij / (dr_norm[:, None] + 1e-10)
        cos_gamma_i = jnp.einsum('aji,ai->aj', pbc_shift(pos[j_list]-ri[:,None,:],box,box_inv)/jnp.linalg.norm(pbc_shift(pos[j_list]-ri[:,None,:],box,box_inv)+1e-10,axis=2,keepdims=True), rij_unit) * j_mask
        cos_gamma_j = jnp.einsum('aji,ai->aj', pbc_shift(pos[k_list]-rj[:,None,:],box,box_inv)/jnp.linalg.norm(pbc_shift(pos[k_list]-rj[:,None,:],box,box_inv)+1e-10,axis=2,keepdims=True), -rij_unit) * k_mask
        ap_feat = self.compute_atompair_features(cos_gamma_i, cos_gamma_j, j_list, k_list, j_mask, k_mask, buffer_nblist_inter_rc, atype_indices, self.apsf_mus, self.apsf_eta)
        ac_feat = (self.compute_atomcenter_features(pos, box, topo_nblist, topo_mask, atype_indices, self.acsf_mus, self.acsf_eta)[pairs[:,0]] + self.compute_atomcenter_features(pos, box, topo_nblist, topo_mask, atype_indices, self.acsf_mus, self.acsf_eta)[pairs[:,1]]) * 0.5
        elem_indices = jnp.array(zindex)[atype_indices]
        atype_oh = jnp.concatenate([jnp.concatenate([elem_indices[pairs[:,0]].reshape(-1,1), int_to_onehot(elem_indices[pairs[:,0]], charge_to_index, 10)], axis=1), jnp.concatenate([elem_indices[pairs[:,1]].reshape(-1,1), int_to_onehot(elem_indices[pairs[:,1]], charge_to_index, 10)], axis=1)], axis=1)
        return jnp.concatenate([ac_feat.reshape(ac_feat.shape[0], -1), ap_feat.reshape(ap_feat.shape[0], -1), atype_oh], axis=1), dr_norm, buffer_nblist_inter_rc

class NeuralNetwork(nn.Module):
    dense_nodes: int = 64
    @nn.compact
    def __call__(self, combined, dr_norm, buffer_nblist_inter):
        x = combined
        for _ in range(3):
            x = nn.relu(nn.LayerNorm()(nn.Dense(self.dense_nodes)(x)))
        return jnp.sum(nn.Dense(1)(x).squeeze(-1) * buffer_nblist_inter)

class EAPNNForceLegacy(nn.Module):
    n_atype: int; rc: float; n_atoms: int; acsf_nmu: int; apsf_nmu: int; acsf_eta: float; apsf_eta: float; use_pbc: bool = True
    def setup(self):
        self.feature_extractor = FeatureExtractor(n_atoms=self.n_atoms, n_atype=self.n_atype, rc=self.rc, acsf_nmu=self.acsf_nmu, apsf_nmu=self.apsf_nmu, acsf_eta=self.acsf_eta, apsf_eta=self.apsf_eta, use_pbc=self.use_pbc)
        self.neural_network = NeuralNetwork()
    def __call__(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        feat, dr, mask = self.feature_extractor(pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
        return self.neural_network(feat, dr, mask)
