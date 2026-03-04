#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import jax
import jax.numpy as jnp
from jax import vmap
from torch.utils.data import DataLoader
from ase.io import read
import glob
from openmm.app import PDBFile
from dmff.api import Hamiltonian
from openmm.unit import angstrom
from openmm.app import CutoffPeriodic
from dmff.common import nblist

# Import project-specific utilities
from phyneo.utils.data_utils import (
    torch_batch_to_jax, setup_plot_style, 
    get_topology_neighbors, filter_and_pad_pairs,
    zindex, MoleculeTorchDataset
)
from phyneo.models.eapnn import EAPNNForce

def main():
    # 1. Configuration
    rc = 6.0
    connectivity = 4
    max_neighbors = 10
    ff_xml = 'phyneo_ecl.xml'
    params_path = 'results/best_model_params.pickle'
    data_path = '../../data/dataset_eapnn/data_all.xyz'
    
    setup_plot_style()
    
    # 2. Load Model and Parameters
    if not os.path.exists(params_path):
        print(f"Error: {params_path} not found. Please train the model first.")
        return

    with open(params_path, 'rb') as f:
        params = pickle.load(f)

    # 3. Load Dataset and Setup Caches (Using test_structures.xyz directly)
    test_path = 'test_structures.xyz'
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found. Please run training script first to generate it.")
        return

    test_structures = read(test_path, ':')
    print(f"Evaluating on {len(test_structures)} test structures from {test_path}")

    # Dimer cache setup
    dimer_file_map = {}
    for pdb_path in glob.glob("../../data/dimer_bank/*.pdb"):
        filename = os.path.basename(pdb_path)
        parts = filename.split('_')
        monomer_A, monomer_B = parts[-2], parts[-1].split('.')[0]
        dimer_file_map[f"{monomer_A}_{monomer_B}"] = pdb_path
        dimer_file_map[f"{monomer_B}_{monomer_A}"] = pdb_path

    nblist_cache = {}
    unique_dimers = set(s.info['Comp'].split(':')[0].split('(')[0] + '_' + 
                        s.info['Comp'].split(':')[1].split('(')[0] 
                        for s in test_structures)

    # Target indices for Li(3) and Na(11)
    LI_ATOMIC_NUM, NA_ATOMIC_NUM = 3.0, 11.0
    li_idx = zindex.index(LI_ATOMIC_NUM)
    na_idx = zindex.index(NA_ATOMIC_NUM)
    TARGET_ATYPE_INDICES = jnp.array([li_idx, na_idx])

    for dimer in unique_dimers:
        if dimer not in dimer_file_map: continue
        pdb_path = dimer_file_map[dimer]
        mol = PDBFile(pdb_path)
        
        # Get atom types for this dimer
        atom_elements = [atom.element.atomic_number for atom in mol.topology.atoms()]
        zindex_dict = {float(num): i for i, num in enumerate(zindex)}
        atype_indices = jnp.array([zindex_dict.get(float(num), -1) for num in atom_elements])
        
        box = jnp.eye(3) * 50 # Large box for dimers
        H = Hamiltonian(ff_xml)
        pots = H.createPotential(mol.topology, nonbondedCutoff=rc*angstrom, 
                                 nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
        nbl = nblist.NoCutoffNeighborList(pots.meta['cov_map'], padding=True)
        pos_dummy = jnp.array(mol.positions._value) * 10
        nbl.allocate(pos_dummy, box)
        pairs = nbl.pairs
        
        # Find target atoms to calculate correct number of pairs
        target_indices = jnp.where((jnp.array(atom_elements) == 3) | (jnp.array(atom_elements) == 11))[0]
        v_pairs, v_mask = filter_and_pad_pairs(pairs, atype_indices, TARGET_ATYPE_INDICES, max_pairs=40)
        t_nblist, t_mask = get_topology_neighbors(pdb_path, connectivity=connectivity, max_neighbors=max_neighbors)
        
        nblist_cache[dimer] = (v_pairs, v_mask, t_nblist, t_mask, atype_indices)

    # Attach cache info to structures
    for structure in test_structures:
        comp = structure.info['Comp']
        m_A, m_B = [c.split('(')[0] for c in comp.split(':')]
        key = f"{m_A}_{m_B}"
        v_pairs, v_mask, t_nblist, t_mask, atypes = nblist_cache[key]
        structure.info.update({'pairs': v_pairs, 'valid_mask': v_mask, 
                               'topo_nblist': t_nblist, 'topo_mask': t_mask})

    # Initialize Model (need n_atoms from a sample)
    sample_structure = test_structures[0]
    n_atoms = len(sample_structure.get_positions())
    model = EAPNNForce(n_atoms=n_atoms, n_atype=len(zindex), rc=rc, 
                       embed_dim=32, n_radial=20, n_angular=12, n_layers=3, hidden_dim=128)

    # 4. Predict on Test Set
    test_dataset = MoleculeTorchDataset(test_structures)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    
    all_ref = []      # Total Reference (QM)
    all_baseline = [] # Baseline FF (sr_energy)
    all_pred_corr = [] # Predicted ML Correction
    all_dist = []

    print("Predicting...")
    for batch in test_dataloader:
        jax_batch = torch_batch_to_jax(batch)
        def predict_fn(sample):
            return model.apply(params, sample['pos'], sample['box'], sample['pairs'], 
                               sample['valid_mask'], sample['topo_nblist'], sample['topo_mask'], 
                               sample['molID'], sample['atypes'])
        
        energy_pred = vmap(predict_fn)(jax_batch)
        
        # User defined: reference = energy + sr_energy
        sr_energy = np.array(jax_batch['sr_energy'])
        energy_qm_diff = np.array(jax_batch['energy'])
        
        all_ref.extend(sr_energy + energy_qm_diff)
        all_baseline.extend(sr_energy)
        all_pred_corr.extend(np.array(energy_pred))
        
        all_dist.extend(np.array(jax_batch['distance']) if 'distance' in jax_batch else [0]*len(energy_pred))

    all_ref = np.array(all_ref)
    all_baseline = np.array(all_baseline)
    all_pred_corr = np.array(all_pred_corr)
    all_corrected = all_baseline + all_pred_corr
    all_dist = np.array(all_dist)

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Parity Plot
    ax1.plot([all_ref.min(), all_ref.max()], [all_ref.min(), all_ref.max()], 'k--', alpha=0.5, label='Perfect (Reference)')
    ax1.scatter(all_ref, all_baseline, color='red', alpha=0.4, label='Baseline (FF)', s=10)
    ax1.scatter(all_ref, all_corrected, color='blue', alpha=0.6, label='Corrected (FF + EAPNN)', s=15)
    
    ax1.set_xlabel('Reference Total Energy (kcal/mol)')
    ax1.set_ylabel('Predicted/Baseline Energy (kcal/mol)')
    ax1.set_title('Parity Plot: Total Energy Comparison')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Residual vs Distance Plot
    res_baseline = all_ref - all_baseline
    res_corrected = all_ref - all_corrected
    
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.scatter(all_dist, res_baseline, color='red', alpha=0.4, label='Ref - Baseline (Target Correction)', s=10)
    ax2.scatter(all_dist, res_corrected, color='blue', alpha=0.6, label='Ref - Corrected (Residual Error)', s=15)
    
    ax2.set_xlabel('Dimer Distance (Å)')
    ax2.set_ylabel('Energy Difference (kcal/mol)')
    ax2.set_title('Residuals vs. Distance')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Calculate RMSE and add to plots
    rmse_baseline = np.sqrt(np.mean(res_baseline**2))
    rmse_corrected = np.sqrt(np.mean(res_corrected**2))
    
    stats_text = f'RMSE Baseline: {rmse_baseline:.3f}\nRMSE Corrected: {rmse_corrected:.3f}'
    # Add to both plots in bottom-right
    for ax in [ax1, ax2]:
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300)
    print("Plot saved as evaluation_results.png")
    
    print(f"\nSTATISTICS:")
    print(f"RMSE Baseline (No Correction): {rmse_baseline:.4f} kcal/mol")
    print(f"RMSE EAPNN (With Correction): {rmse_corrected:.4f} kcal/mol")
    print(f"Error Reduction: {(1 - rmse_corrected/rmse_baseline)*100:.2f}%")

if __name__ == "__main__":
    main()
