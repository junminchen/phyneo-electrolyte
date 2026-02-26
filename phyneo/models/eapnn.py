#!/usr/bin/env python
"""
Environment-Aware Pairwise Neural Network (EAPNN) for Potential Correction.

This module implements an optimized neural network architecture to learn 
corrections to physics-based pairwise potentials with enhanced descriptive 
power and transferability.
"""

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
from flax import linen as nn
from typing import Tuple, Dict, List, Any, Optional
from functools import partial

from dmff.utils import jit_condition, regularize_pairs, pair_buffer_scales
from dmff.admp.pairwise import distribute_scalar
from dmff.admp.spatial import pbc_shift
from dmff.common import nblist

# Import utility functions
from phyneo.utils.data_utils import (
    cutoff_cosine, zindex
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

class BesselBasis(nn.Module):
    """
    Bessel Radial Basis Functions for better radial feature representation.
    """
    n_basis: int
    rc: float

    def setup(self):
        self.frequencies = jnp.pi * jnp.arange(1, self.n_basis + 1) / self.rc

    def __call__(self, x):
        # x: [..., 1]
        return jnp.sqrt(2 / self.rc) * jnp.sin(self.frequencies * x) / (x + 1e-10)

class AngularBasis(nn.Module):
    """
    Angular Basis Functions (Legendre-like or Gaussian on Cosine).
    """
    n_basis: int
    
    def setup(self):
        self.mus = jnp.linspace(-1.0, 1.0, self.n_basis)
        self.eta = 2.0 / (self.mus[1] - self.mus[0])**2

    def __call__(self, cos_theta):
        # cos_theta: [..., 1]
        return jnp.exp(-self.eta * jnp.square(cos_theta - self.mus))

class ResidualBlock(nn.Module):
    """
    Standard Residual Block for MLP.
    """
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
    """
    Attention-based interaction to aggregate environmental information.
    """
    embed_dim: int
    num_heads: int = 4

    @nn.compact
    def __call__(self, query, key_value, mask):
        # query: [batch, 1, dim]
        # key_value: [batch, neighbors, dim]
        # mask: [batch, neighbors]
        
        # Multi-head Attention
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            deterministic=True,
            dtype=jnp.float32
        )(query, key_value, mask=mask[:, None, None, :])
        
        return jnp.squeeze(attn_out, axis=1)

class EAPNNForce(nn.Module):
    """
    Optimized Environment-Aware Pairwise Neural Network.
    """
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
        # Atomic Embeddings
        self.embedding = nn.Embed(num_embeddings=self.n_atype + 1, features=self.embed_dim)
        
        # Basis Functions
        self.radial_basis = BesselBasis(n_basis=self.n_radial, rc=self.rc)
        self.angular_basis = AngularBasis(n_basis=self.n_angular)
        
        # Interaction aggregation
        self.attention = AttentionInteraction(embed_dim=self.embed_dim)
        self.dense_v_env = nn.Dense(self.embed_dim)
        self.dense_q = nn.Dense(self.embed_dim)
        
        # Final MLP
        self.dense_combined = nn.Dense(self.hidden_dim)
        layers = []
        for _ in range(self.n_layers):
            layers.append(ResidualBlock(features=self.hidden_dim))
        self.mlp = nn.Sequential(layers)
        self.final_dense = nn.Dense(1)

    def _compute_internal(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices, train: bool = False):
        # 1. Geometry processing
        box_inv = jnp.linalg.inv(box)
        
        # Pair processing
        pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
        # Note: we use buffer_scales to filter valid interactions and apply cutoffs
        buffer_scales = pair_buffer_scales(pairs[:, :2]) * valid_mask
        
        ri = pos[pairs[:, 0]]
        rj = pos[pairs[:, 1]]
        rij = pbc_shift(rj - ri, box, box_inv)
        dr_norm = jnp.linalg.norm(rij + 1e-10, axis=1, keepdims=True)
        rij_unit = rij / dr_norm
        
        # Intermolecular mask & Cutoff
        same_mol = mol_ID[pairs[:, 0]] == mol_ID[pairs[:, 1]]
        buffer_inter = jnp.where(same_mol, 0., 1.)
        f_cut = cutoff_cosine(dr_norm, self.rc)
        mask_final = buffer_inter * buffer_scales[:, None] * f_cut
        
        # 2. Embedding
        # Handle padding index for embedding (map 10000 to n_atype)
        safe_atypes = jnp.where(atype_indices == 10000, self.n_atype, atype_indices)
        h = self.embedding(safe_atypes) # [n_atoms, embed_dim]
        hi = h[pairs[:, 0]]
        hj = h[pairs[:, 1]]
        
        # 3. Environment processing
        j_list, k_list, j_mask, k_mask = get_environment_atoms(pairs, topo_nblist, topo_mask)
        
        # Process environment of i
        ri_env = pos[j_list]
        ri_X = pbc_shift(ri_env - ri[:, None, :], box, box_inv)
        di_X_norm = jnp.linalg.norm(ri_X + 1e-10, axis=2, keepdims=True)
        cos_gamma_i = jnp.einsum('ajk,ak->aj', ri_X / di_X_norm, rij_unit)
        
        # Environment features for i
        feat_i_radial = self.radial_basis(di_X_norm) # [n_pairs, n_neighbors, n_radial]
        feat_i_angular = self.angular_basis(cos_gamma_i[..., None]) # [n_pairs, n_neighbors, n_angular]
        feat_i_env = jnp.concatenate([feat_i_radial, feat_i_angular], axis=-1)
        
        # Neighbor embeddings
        h_env_i = h[j_list] # [n_pairs, n_neighbors, embed_dim]
        # Combine geometry and chemical info
        v_env_i = self.dense_v_env(feat_i_env) * h_env_i
        
        # Aggregate i-environment using attention
        q_i = self.dense_q(hi[:, None, :])
        env_agg_i = self.attention(q_i, v_env_i, j_mask) # [n_pairs, embed_dim]
        
        # Process environment of j (symmetric to i)
        rj_env = pos[k_list]
        rj_X = pbc_shift(rj_env - rj[:, None, :], box, box_inv)
        dj_X_norm = jnp.linalg.norm(rj_X + 1e-10, axis=2, keepdims=True)
        cos_gamma_j = jnp.einsum('ajk,ak->aj', rj_X / dj_X_norm, -rij_unit)
        
        feat_j_radial = self.radial_basis(dj_X_norm)
        feat_j_angular = self.angular_basis(cos_gamma_j[..., None])
        feat_j_env = jnp.concatenate([feat_j_radial, feat_j_angular], axis=-1)
        
        h_env_j = h[k_list]
        v_env_j = self.dense_v_env(feat_j_env) * h_env_j
        
        q_j = self.dense_q(hj[:, None, :])
        env_agg_j = self.attention(q_j, v_env_j, k_mask)
        
        # 4. Pair interaction feature
        radial_ij = self.radial_basis(dr_norm) # [n_pairs, n_radial]
        
        # Combined pair features: [center_i, center_j, distance_ij, env_i, env_j]
        combined = jnp.concatenate([
            hi, hj, radial_ij, env_agg_i, env_agg_j
        ], axis=-1)
        
        return combined, dr_norm, mask_final

    def __call__(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices, train: bool = False):
        combined, dr_norm, mask_final = self._compute_internal(pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices, train=train)
        
        # 5. Predict Energy
        x = self.dense_combined(combined)
        x = self.mlp(x, train=train)
        out_pair = self.final_dense(x)
        
        # Apply mask and sum
        energy = jnp.sum(out_pair * mask_final)
        return energy

    def get_features(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices, train: bool = False):
        """Method to extract intermediate features for debugging/analysis"""
        combined, dr_norm, mask_final = self._compute_internal(pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices, train=train)
        return combined, dr_norm, mask_final

    def predict_energy_force(self, params, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        """Simultaneously predict energy and force"""
        def energy_fn(p):
            return self.apply(
                params, p, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices
            )
        
        energy, grad_energy = value_and_grad(energy_fn)(pos)
        return energy, -grad_energy # force = -grad E
