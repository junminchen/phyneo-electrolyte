#!/usr/bin/env python
"""
Training script for Li pair interactions using EAPNN model.
This script trains a neural network force field for Li-containing molecular pairs.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import optax
import pickle

from openmm.app import PDBFile
from openmm.unit import angstrom
from openmm.app import CutoffPeriodic

from dmff.api import Hamiltonian
from dmff.common import nblist

# Import custom modules (assumed to be in phyneo package)
from phyneo.models.eapnn import EAPNNForce
from phyneo.data.dataset import MoleculeTorchDataset
from phyneo.training.utils import calculate_weights, filter_and_pad_pairs, get_topology_neighbors

def create_train_state(model, learning_rate, key, init_batch):
    """Initialize training state with optimizer."""
    init_pos = init_batch['pos'][0]
    init_box = init_batch['box'][0]
    init_pairs = init_batch['pairs'][0]
    init_valid_mask = init_batch['valid_mask'][0]
    init_topo_nblist = init_batch['topo_nblist'][0]
    init_topo_mask = init_batch['topo_mask'][0]
    init_molID = init_batch['molID'][0]
    init_atypes = init_batch['atypes'][0]
    
    params = model.init(
        key, init_pos, init_box, init_pairs, init_valid_mask, 
        init_topo_nblist, init_topo_mask, init_molID, init_atypes
    )
    
    tx = optax.adam(learning_rate=learning_rate)
    return params, tx

@jit
def train_step(params, opt_state, batch, optimizer, force_weight=10.0):
    """Single training step."""
    def loss_fn(params):
        # Predict energy and forces
        def predict_single(sample):
            energy_pred, force_pred = model.predict_energy_force(
                params,
                pos=sample['pos'],
                box=sample['box'],
                pairs=sample['pairs'],
                valid_mask=sample['valid_mask'],
                topo_nblist=sample['topo_nblist'],
                topo_mask=sample['topo_mask'],
                mol_ID=sample['molID'],
                atype_indices=sample['atypes']
            )
            return energy_pred, force_pred
        
        energy_pred, force_pred = vmap(predict_single)(batch)
        
        # Energy loss
        energy_true = batch['energy']
        energy_loss = jnp.mean((energy_pred - energy_true) ** 2)
        
        # Force loss with masking
        force_true = batch['forces']
        atom_mask = batch['atom_mask'][..., None]
        force_error = (force_pred - force_true) * atom_mask
        force_loss = jnp.mean(force_error ** 2)
        
        # Total loss
        total_loss = energy_loss + force_weight * force_loss
        return total_loss, (energy_loss, force_loss)
    
    (total_loss, (energy_loss, force_loss)), grads = value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, total_loss, energy_loss, force_loss

def evaluate(model, params, test_dataloader):
    """Evaluate model on test set."""
    energy_rmse_list = []
    force_rmse_list = []
    
    for batch in test_dataloader:
        batch = torch_batch_to_jax(batch)
        
        def predict_single(sample):
            energy_pred, force_pred = model.predict_energy_force(
                params,
                pos=sample['pos'],
                box=sample['box'],
                pairs=sample['pairs'],
                valid_mask=sample['valid_mask'],
                topo_nblist=sample['topo_nblist'],
                topo_mask=sample['topo_mask'],
                mol_ID=sample['molID'],
                atype_indices=sample['atypes']
            )
            return energy_pred, force_pred
        
        energy_pred, force_pred = vmap(predict_single)(batch)
        
        # Energy RMSE
        energy_true = batch['energy']
        energy_rmse = jnp.sqrt(jnp.mean((energy_pred - energy_true) ** 2))
        energy_rmse_list.append(energy_rmse)
        
        # Force RMSE
        force_true = batch['forces']
        atom_mask = batch['atom_mask'][..., None]
        force_error = (force_pred - force_true) * atom_mask
        force_rmse = jnp.sqrt(jnp.mean(force_error ** 2))
        force_rmse_list.append(force_rmse)
    
    return {
        'energy_rmse': jnp.mean(jnp.array(energy_rmse_list)),
        'force_rmse': jnp.mean(jnp.array(force_rmse_list))
    }

def main():
    # Configuration
    rc = 6.0
    connectivity = 4
    max_neighbors = 10
    acsf_nmu = 20
    apsf_nmu = 20
    acsf_eta = 100
    apsf_eta = 50
    
    # Model parameters
    learning_rate = 1e-3
    n_epochs = 5000
    batch_size = 64
    
    # Data paths
    ff_xml = 'data/force_fields/output.xml'
    data_file = 'data/processed/data_li_pairs.xyz'
    output_dir = 'results/li_pairs'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    from ase.io import read
    ase_structures = read(data_file, ':')
    
    # Split train/test
    import random
    random.seed(1234)
    random.shuffle(ase_structures)
    train_structures = ase_structures[:int(0.9*len(ase_structures))]
    test_structures = ase_structures[int(0.9*len(ase_structures)):]
    
    # Create datasets
    train_dataset = MoleculeTorchDataset(train_structures)
    test_dataset = MoleculeTorchDataset(test_structures)
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, drop_last=False)
    
    # Initialize model
    # (Model initialization code here)
    
    # Training loop
    print("Starting training...")
    for epoch in range(n_epochs):
        # Training code here
        pass
    
    print("Training complete!")


if __name__ == "__main__":
    main()