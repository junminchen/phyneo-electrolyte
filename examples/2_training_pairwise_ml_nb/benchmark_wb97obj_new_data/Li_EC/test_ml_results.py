#!/usr/bin/env python
import os
import sys
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import mdtraj as md
from openmm.app import PDBFile
from dmff.api import Hamiltonian
from openmm.unit import angstrom
from openmm.app import CutoffPeriodic
from dmff.common import nblist
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath("../../../../"))

from phyneo.models.eapnn import EAPNNForce
from phyneo.utils.data_utils import get_topology_neighbors, zindex, filter_and_pad_pairs, setup_plot_style

# Custom unpickler to fix numpy compatibility
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core.multiarray': module = 'numpy.core.multiarray'
        if module == 'numpy._core.numeric': module = 'numpy.core.numeric'
        return super().find_class(module, name)

def main():
    # 1. Configuration
    rc = 6.0
    connectivity = 4
    max_neighbors = 10
    
    # Target indices for Li(3) and Na(11)
    li_idx = zindex.index(3.0)
    na_idx = zindex.index(11.0)
    TARGET_ATYPE_INDICES = jnp.array([li_idx, na_idx])

    params_path = '../../results/best_model_params.pickle'
    data_pickle_path = 'data_wb97obj_force_wt_ff.pickle' # Use the one WITH pre-calculated FF
    pdb_path = '../../dimer_062_Li_EC.pdb'
    
    print(f"Loading parameters from {params_path}...")
    with open(params_path, 'rb') as f:
        try:
            params_dict = pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            params_dict = CustomUnpickler(f).load()
            
    params = params_dict.get('params', params_dict)
    if 'params' not in params:
        params = {'params': params}

    print(f"Loading data from {data_pickle_path}...")
    with open(data_pickle_path, 'rb') as f:
        data = pickle.load(f)
    scan_res = data['conf_062_Li_EC']

    # 2. Setup Model and Topology
    print("Setting up model and topology...")
    mol = PDBFile(pdb_path)
    box = jnp.array(mol.topology.getPeriodicBoxVectors()._value) * 10
    mol_ID = jnp.array([atom.residue.index for atom in mol.topology.atoms()])
    atom_elements = [atom.element.atomic_number for atom in mol.topology.atoms()]
    zindex_dict = {float(num): i for i, num in enumerate(zindex)}
    atype_indices = jnp.array([zindex_dict.get(float(num), -1) for num in atom_elements])
    n_atoms = len(mol.positions)

    topo_nblist, topo_mask = get_topology_neighbors(pdb_path, connectivity=connectivity, max_neighbors=max_neighbors)

    # Setup Neighbor List for ML model input structure
    H = Hamiltonian('../../phyneo_ecl.xml')
    pots = H.createPotential(mol.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
    nbl = nblist.NoCutoffNeighborList(pots.meta['cov_map'], padding=True)
    pos_dummy = jnp.array(mol.positions._value) * 10
    nbl.allocate(pos_dummy, box)
    pairs = nbl.pairs
    valid_pairs, valid_mask = filter_and_pad_pairs(pairs, atype_indices, TARGET_ATYPE_INDICES, max_pairs=40)

    model = EAPNNForce(
        n_atoms=n_atoms, n_atype=len(zindex), rc=rc,  
        embed_dim=32, n_radial=20, n_angular=12, n_layers=3, hidden_dim=128,
        use_pbc=True
    )

    # 3. Predict ML Corrections
    print("Predicting ML corrections...")
    calc_func = jit(model.predict_energy_force)
    
    ene_ml_corrs = []
    force_ml_corrs = []
    
    for i in range(len(scan_res['posA'])):
        pos = jnp.concatenate([scan_res['posA'][i], scan_res['posB'][i]], axis=0)
        energy, force = calc_func(params, pos, box, valid_pairs, valid_mask, 
                                 topo_nblist, topo_mask, mol_ID, atype_indices)
        ene_ml_corrs.append(energy)
        force_ml_corrs.append(force)

    ene_ml_corrs = np.array(ene_ml_corrs)
    force_ml_corrs = np.array(force_ml_corrs)

    # 4. Aligned Comparison
    ene_dft = np.array(scan_res['tot'])
    ene_ff = np.array(scan_res['ene_ff'])
    
    # ML model learns residual (E_DFT - E_FF).
    # Since there's a baseline shift in new data, we align the correction to the long-range part.
    # Typically at large distances, the correction should go to a constant or 0.
    target_residual = ene_dft - ene_ff
    shift = np.mean(target_residual[-5:]) - np.mean(ene_ml_corrs[-5:])
    
    # Aligned Correction
    ene_ml_aligned = ene_ml_corrs + shift
    ene_corrected = ene_ff + ene_ml_aligned
    
    rmse_ff = np.sqrt(np.mean((ene_dft - ene_ff)**2))
    rmse_corr = np.sqrt(np.mean((ene_dft - ene_corrected)**2))

    print(f"\nCorrection Alignment Shift: {shift:.4f} kJ/mol")
    print(f"Energy RMSE (Baseline FF): {rmse_ff:.4f} kJ/mol")
    print(f"Energy RMSE (Corrected):   {rmse_corr:.4f} kJ/mol")

    # Forces
    force_dft = np.array(scan_res['grad'])
    force_ff = np.array(scan_res['force_ff'])
    force_corrected = force_ff + force_ml_corrs
    
    rmse_f_ff = np.sqrt(np.mean((force_dft - force_ff)**2))
    rmse_f_corr = np.sqrt(np.mean((force_dft - force_corrected)**2))
    
    print(f"Force RMSE (Baseline FF):  {rmse_f_ff:.4f} kJ/mol/A")
    print(f"Force RMSE (Corrected):    {rmse_f_corr:.4f} kJ/mol/A")

    # 5. Plotting
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy Plot
    ax1.scatter(ene_dft, ene_ff, label=f'FF (RMSE: {rmse_ff:.2f})', alpha=0.5, color='red')
    ax1.scatter(ene_dft, ene_corrected, label=f'Corrected (RMSE: {rmse_corr:.2f})', alpha=0.6, color='blue')
    ax1.plot([ene_dft.min(), ene_dft.max()], [ene_dft.min(), ene_dft.max()], 'k--')
    ax1.set_xlabel('DFT Energy (kJ/mol)')
    ax1.set_ylabel('Predicted Energy (kJ/mol)')
    ax1.set_title('Energy Parity (Shift Aligned)')
    ax1.legend()

    # Residual vs Distance
    # Distances between Li and first atom of EC
    distances = np.linalg.norm(scan_res['posA'][:, 0, :] - scan_res['posB'][:, 0, :], axis=1)
    ax2.plot(distances, target_residual, 'r-', label='Target (DFT - FF)')
    ax2.plot(distances, ene_ml_aligned, 'b--', label='ML Predicted (Aligned)')
    ax2.set_xlabel('Li-EC Distance (A)')
    ax2.set_ylabel('Correction (kJ/mol)')
    ax2.set_title('Correction Curve')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('ml_results_aligned.png', dpi=300)
    print("\nEvaluation complete. Plot saved to ml_results_aligned.png")

if __name__ == "__main__":
    main()
