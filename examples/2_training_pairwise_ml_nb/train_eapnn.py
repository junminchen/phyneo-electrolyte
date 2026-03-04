#!/usr/bin/env python
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
from functools import partial
# config.update("jax_enable_x64", True)  # Enable for higher precision if needed
config.update("jax_debug_nans", True)

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

import MDAnalysis as mda
from ase.io import read, write
import mdtraj as md
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Any
from scipy.sparse import csr_matrix

# Import utility functions
from phyneo.utils.data_utils import (
    torch_batch_to_jax, setup_plot_style, plot_training_progress,
    get_topology_neighbors, filter_and_pad_pairs, cutoff_cosine,
    int_to_onehot, get_data, parameter_shapes, zindex, charge_to_index, MoleculeTorchDataset
)
from phyneo.models.eapnn import EAPNNForce

# Target atom type indices (Li and Na)
LI_ATOMIC_NUM = 3.0    # Atomic number of Li
NA_ATOMIC_NUM = 11.0   # Atomic number of Na
li_index = zindex.index(LI_ATOMIC_NUM) if LI_ATOMIC_NUM in zindex else -1
na_index = zindex.index(NA_ATOMIC_NUM) if NA_ATOMIC_NUM in zindex else -1
TARGET_ATYPE_INDICES = jnp.array([li_index, na_index])

if __name__ == "__main__":
    rc = 6.0
    connectivity = 4
    max_neighbors = 10

    ff_xml = 'phyneo_ecl.xml'
    pdb = 'dimer_062_Li_EC.pdb'

    outfile = '../../data/dataset_eapnn/data_all.xyz'

    # Set up force field
    mol = PDBFile(pdb)
    pos = jnp.array(mol.positions._value) * 10
    box = jnp.array(mol.topology.getPeriodicBoxVectors()._value) * 10

    H = Hamiltonian(ff_xml)
    pots = H.createPotential(mol.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)

    nbl = nblist.NoCutoffNeighborList(pots.meta['cov_map'], padding=True)
    nbl.capacity_multiplier = 1000
    nbl.allocate(pos, box)
    pairs = nbl.pairs

    mol_ID = []
    for atom in mol.topology.atoms():
        mol_ID.append(atom.residue.index)
    mol_ID = jnp.array(mol_ID)

    atom_elements = []
    for atom in mol.topology.atoms():
        atom_elements.append(atom.element.atomic_number)
    z_atomnum = jnp.array(atom_elements)

    z_atomnum_list = [float(num) for num in np.array(z_atomnum)]
    zindex_dict = {float(num): i for i, num in enumerate(zindex)}
    atype_indices = jnp.array([zindex_dict.get(num, -1) for num in z_atomnum_list])
    n_atoms = len(pos)

    # Extract atomic numbers
    atomic_nums = jnp.array([atom.element.atomic_number for atom in mol.topology.atoms()], dtype=int)
    # Mark Li(3) and Na(11) atoms
    target_mask = (atomic_nums == 3) | (atomic_nums == 11)
    target_indices = jnp.where(target_mask)[0]

    valid_pairs, valid_mask = filter_and_pad_pairs(pairs, atype_indices, TARGET_ATYPE_INDICES, max_pairs=len(target_indices)*100)

    # Get topology neighbors
    topo_nblist, topo_mask = get_topology_neighbors(pdb, connectivity=connectivity, max_neighbors=max_neighbors, max_n_atoms=None)

    model = EAPNNForce(
        n_atoms=n_atoms, 
        n_atype=len(zindex), 
        rc=rc,  
        embed_dim=32,
        n_radial=20,
        n_angular=12,
        n_layers=3,
        hidden_dim=128,
        use_pbc=True,
    )

    key = jax.random.PRNGKey(0)

    params_init = model.init(key, pos, box, valid_pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
    start_time = time.time()

    params = params_init
    # Test forward pass
    energy = model.apply(params, pos, box, valid_pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
    print(f"Initial energy: {energy}")
    end_time = time.time()
    print(f"time cost: {end_time - start_time}s")

    def dmff_calculator(pos, box, pairs, valid_pairs, valid_mask):
        E_nb_ml = model.apply(params, pos, box, valid_pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
        return E_nb_ml
    calc_dmff = jit(value_and_grad(dmff_calculator,argnums=(0, 1)))
    # compile tot_force function
    energy, (grad, virial) = calc_dmff(pos, box, pairs, valid_pairs, valid_mask)
    print(f"Compiled energy/grad: {energy}, {grad.shape}, {virial.shape}")

    num_runs = 10
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        _ = calc_dmff(pos, box, pairs, valid_pairs, valid_mask)
        end_time = time.time()
        total_time += (end_time - start_time)

    average_time = total_time / num_runs
    print(f"Average calculation time: {average_time:.4f} seconds")

    data_ase = outfile
    ase_structures = read(data_ase, ':')
    dimer_file_map = {}

    for pdb_path in glob.glob("../../data/dimer_bank/*.pdb"):
        filename = os.path.basename(pdb_path)
        parts = filename.split('_')
        monomer_A, monomer_B = parts[-2], parts[-1].split('.')[0]
        dimer_file_map[f"{monomer_A}_{monomer_B}"] = pdb_path
        dimer_file_map[f"{monomer_B}_{monomer_A}"] = pdb_path

    unique_dimers = set(structure.info['Comp'].split(':')[0].split('(')[0] + '_' + 
                        structure.info['Comp'].split(':')[1].split('(')[0] 
                        for structure in ase_structures)

    nblist_cache = {}
    for dimer in unique_dimers:
        monomer_A, monomer_B = dimer.split('_')
        if dimer not in dimer_file_map:
            continue
        
        pdb_path = dimer_file_map[dimer]
        mol = PDBFile(pdb_path)
        box = jnp.eye(3) * 50
        H = Hamiltonian(ff_xml)
        pots = H.createPotential(
            mol.topology,
            nonbondedCutoff=rc*angstrom,
            nonbondedMethod=CutoffPeriodic,
            ethresh=1e-4
        )
        cov_map = pots.meta['cov_map']
        
        pos_dummy = jnp.array(mol.positions._value)
        nbl = nblist.NoCutoffNeighborList(pots.meta['cov_map'], padding=True)
        nbl.capacity_multiplier = 800
        pairs_extracted = nbl.allocate(pos_dummy, box)   

        pairs_extracted, valid_mask_extracted = filter_and_pad_pairs(pairs_extracted, atype_indices, TARGET_ATYPE_INDICES, max_pairs=40)
        topo_nblist_extracted, topo_mask_extracted = get_topology_neighbors(pdb_path, connectivity=connectivity, max_neighbors=max_neighbors, max_n_atoms=None)
        
        nblist_cache[dimer] = (pairs_extracted, valid_mask_extracted, topo_nblist_extracted, topo_mask_extracted)

    print(f"\nDATASET ANALYSIS:")
    print(f"Total structures: {len(ase_structures)}")
    print(f"Found PDB files for: {len(unique_dimers)} dimer types")

    import random
    random.seed(1234)
    random.shuffle(ase_structures)
    train_structures = ase_structures[:int(0.9*len(ase_structures))]
    test_structures = ase_structures[int(0.9*len(ase_structures)):]
    write('test_structures.xyz', test_structures)

    def prepare_structures(structures):
        for structure in structures:
            comp = structure.info['Comp']
            monomer_A, monomer_B = comp.split(':')
            monomer_A = monomer_A.split('(')[0]
            monomer_B = monomer_B.split('(')[0]
            key = f"{monomer_A}_{monomer_B}"
            if key not in nblist_cache:
                continue
            pairs_cache, valid_mask_cache, topo_nblist_cache, topo_mask_cache = nblist_cache[key]
            structure.info['pairs'] = pairs_cache
            structure.info['valid_mask'] = valid_mask_cache
            structure.info['topo_nblist'] = topo_nblist_cache
            structure.info['topo_mask'] = topo_mask_cache

    prepare_structures(train_structures)
    prepare_structures(test_structures)
            
    train_dataset = MoleculeTorchDataset(train_structures)
    test_dataset = MoleculeTorchDataset(test_structures)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, drop_last=False)

    try:
        batch = next(iter(train_dataloader))
        print("Successfully loaded a batch:")
        for k, v in batch.items():
            print(f"{k}: shape {jnp.array(v).shape}")
    except Exception as e:
        print(f"Error occurred during dataloading: {e}")
        
    batch = torch_batch_to_jax(batch)

    def create_train_state(model, learning_rate, key, init_batch):
        init_pos, init_box = init_batch['pos'][0], init_batch['box'][0]
        init_pairs, init_valid_mask = init_batch['pairs'][0], init_batch['valid_mask'][0]
        init_topo_nblist, init_topo_mask = init_batch['topo_nblist'][0], init_batch['topo_mask'][0]
        init_molID, init_atypes = init_batch['molID'][0], init_batch['atypes'][0]
        
        params = model.init(key, init_pos, init_box, init_pairs, init_valid_mask, init_topo_nblist, init_topo_mask, init_molID, init_atypes)
        tx = optax.adam(learning_rate=learning_rate)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @partial(jit, static_argnums=(2,))
    def train_step(state, batch, force_weight=0):
        def loss_fn(params):
            if force_weight == 0:
                def predict_energy(sample):
                    return model.apply(params, sample['pos'], sample['box'], sample['pairs'], sample['valid_mask'], sample['topo_nblist'], sample['topo_mask'], sample['molID'], sample['atypes'], train=True)
                energy_pred = vmap(predict_energy)(batch)
                energy_true = batch['energy']
                energy_loss = jnp.mean((energy_pred - energy_true) ** 2)
                return energy_loss, (energy_loss, jnp.array(0.0))
            else:
                def predict_energy_force(sample):
                    return model.predict_energy_force(params, sample['pos'], sample['box'], sample['pairs'], sample['valid_mask'], sample['topo_nblist'], sample['topo_mask'], sample['molID'], sample['atypes'])
                energy_pred, force_pred = vmap(predict_energy_force)(batch)
                energy_true = batch['energy']
                energy_loss = jnp.mean((energy_pred - energy_true) ** 2)
                force_true = batch['forces']
                atom_mask = batch['atom_mask'][..., None]
                force_error = (force_pred - force_true) * atom_mask
                force_loss = jnp.mean(force_error ** 2)
                total_loss = energy_loss + force_weight * force_loss
                return total_loss, (energy_loss, force_loss)
        
        (total_loss, (energy_loss, force_loss)), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, total_loss, energy_loss, force_loss

    def evaluate(model, params, test_dataloader, force_weight=0):
        energy_rmse_list = []
        force_rmse_list = []
        
        for batch_eval in test_dataloader:
            batch_eval = torch_batch_to_jax(batch_eval)
            if force_weight == 0:
                def predict_energy(sample):
                    return model.apply(params, sample['pos'], sample['box'], sample['pairs'], sample['valid_mask'], sample['topo_nblist'], sample['topo_mask'], sample['molID'], sample['atypes'], train=False)
                energy_pred = vmap(predict_energy)(batch_eval)
                energy_true = batch_eval['energy']
                energy_rmse = jnp.sqrt(jnp.mean((energy_pred - energy_true) ** 2))
                energy_rmse_list.append(energy_rmse)
                force_rmse_list.append(jnp.array(0.0))
            else:
                def predict_energy_force(sample):
                    return model.predict_energy_force(params, sample['pos'], sample['box'], sample['pairs'], sample['valid_mask'], sample['topo_nblist'], sample['topo_mask'], sample['molID'], sample['atypes'])
                energy_pred, force_pred = vmap(predict_energy_force)(batch_eval)
                energy_true = batch_eval['energy']
                energy_rmse = jnp.sqrt(jnp.mean((energy_pred - energy_true) ** 2))
                energy_rmse_list.append(energy_rmse)
                force_true = batch_eval['forces']
                atom_mask = batch_eval['atom_mask'][..., None]
                force_error = (force_pred - force_true) * atom_mask
                force_rmse = jnp.sqrt(jnp.mean(force_error ** 2))
                force_rmse_list.append(force_rmse)
        
        return {
            'energy_rmse': jnp.mean(jnp.array(energy_rmse_list)),
            'force_rmse': jnp.mean(jnp.array(force_rmse_list))
        }

    learning_rate = 1e-3
    state = create_train_state(model, learning_rate, jax.random.PRNGKey(0), batch)

    print("Starting training...")
    start_time = time.time()
    train_metrics = {'total_loss': [], 'energy_loss': [], 'force_loss': []}
    test_metrics = {'energy_rmse': [], 'force_rmse': []}
    num_epochs = 5000
    force_weight = 0 # Can be set to non-zero to include forces
    
    # Early stopping parameters
    patience = 50  # Stop if no improvement after 50 evaluations (500 epochs)
    best_energy_rmse = float('inf')
    epochs_no_improve = 0
    best_params = None

    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_total_loss, train_energy_loss, train_force_loss = [], [], []
        
        for batch_train in train_dataloader:
            batch_train = torch_batch_to_jax(batch_train)
            state, total_loss, energy_loss, force_loss = train_step(state, batch_train, force_weight)
            train_total_loss.append(total_loss.item())
            train_energy_loss.append(energy_loss.item())
            train_force_loss.append(force_loss.item())
        
        if epoch % 10 == 0:
            avg_total_loss = np.mean(train_total_loss)
            avg_energy_loss = np.mean(train_energy_loss)
            avg_force_loss = np.mean(train_force_loss)
            train_metrics['total_loss'].append(avg_total_loss)
            train_metrics['energy_loss'].append(avg_energy_loss)
            train_metrics['force_loss'].append(avg_force_loss)
            
            test_metric = evaluate(model, state.params, test_dataloader, force_weight)
            current_energy_rmse = test_metric['energy_rmse'].item()
            test_metrics['energy_rmse'].append(current_energy_rmse)
            test_metrics['force_rmse'].append(test_metric['force_rmse'].item())
            
            epoch_time = time.time() - epoch_start
            total_time_elapsed = time.time() - start_time
            clear_output(wait=True)
            print(f"Epoch {epoch:4d}/{num_epochs} | "
                  f"Total Loss: {avg_total_loss:.4f} | "
                  f"Energy Loss: {avg_energy_loss:.4f} | "
                  f"Force Loss: {avg_force_loss:.4f} | "
                  f"Test Energy RMSE: {current_energy_rmse:.4f} | "
                  f"Test Force RMSE: {test_metric['force_rmse']:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            fig = plot_training_progress(epoch, num_epochs, train_metrics, test_metrics, total_time_elapsed)
            plt.savefig(f"results/training_progress_epoch_{epoch}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            os.makedirs("results", exist_ok=True)
            # Check for improvement
            if current_energy_rmse < best_energy_rmse:
                best_energy_rmse = current_energy_rmse
                epochs_no_improve = 0
                best_params = state.params
                with open(f"results/best_model_params.pickle", 'wb') as f:
                    pickle.dump(best_params, f)
                print(f"New best model saved with Energy RMSE: {best_energy_rmse:.4f}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} evaluations. (Best: {best_energy_rmse:.4f})")
            
            with open(f"results/model_params_epoch_{epoch}.pickle", 'wb') as f:
                pickle.dump(state.params, f)
            
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print("Training complete!")
    if best_params is not None:
        print(f"Saving final model using best parameters from epoch {epoch - epochs_no_improve * 10}")
        with open("results/final_model_params.pickle", 'wb') as f:
            pickle.dump(best_params, f)
    else:
        with open("results/final_model_params.pickle", 'wb') as f:
            pickle.dump(state.params, f)
            
    with open("results/training_metrics.pickle", 'wb') as f:
        pickle.dump({'train': train_metrics, 'test': test_metrics}, f)
