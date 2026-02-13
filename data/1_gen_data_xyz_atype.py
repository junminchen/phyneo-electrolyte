#!/usr/bin/env python
"""
Molecular Data Generation Script
Used to generate training data in XYZ format from molecular dynamics trajectory data.
Includes atom type information and intermolecular interaction energy.

Performance Optimized Version:
1. Uses JAX vectorized calculations
2. Caches molecular data to avoid redundant loading
3. Processes data in batches
4. Parallel file writing
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import pickle
import os
from typing import List, Tuple, Dict, Any
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# Scientific Computing Library Imports
# =============================================================================
import numpy as np
import mdtraj as md

# =============================================================================
# JAX Related Imports
# =============================================================================
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.config import config

# Enable JAX optimization
# config.update('jax_enable_x64', True)

# =============================================================================
# Molecular Dynamics Library Imports
# =============================================================================
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *
from dmff.api import Hamiltonian

# =============================================================================
# Global Configuration
# =============================================================================
# File path configuration
FF_XML = 'phyneo_ecl.xml'
DATA_FILE = 'data_dimer_wt_ff.pickle'

OUTFILE = 'dataset_eapnn/data_all.xyz'

# Molecular type definitions
SOLVENTS = ['DEC', 'DMC', 'DME', 'EC', 'EMC', 'DOL', 'DFEA', 'DFEC', 'EP',
            'FEC', 'FEMC', 'GBL', 'PC', 'PP', 'PS', 'SL']
IONS = ['Li', 'Na', 'PF6', 'BOB', 'FSI', 'TFSI', 'BF4', 'DFP', 'DFOB']
CATIONS = ['CN1', 'CN2']

# Energy thresholds
# Define thresholds for outlier filtering (adjust according to actual data!)
ENERGY_LOWER = -500.0
ENERGY_UPPER = 500.0
DISTANCE_LOWER = 1.0
DISTANCE_UPPER = 9.0
# Added: reasonable range for target_tot_full_energy (total energy deviation threshold)
TOT_FULL_ENERGY_LOWER = -1000.0  # Example: lower bound of total energy
TOT_FULL_ENERGY_UPPER = 1000.0   # Example: upper bound of total energy


# Performance configuration
BATCH_SIZE = 1000  # Batch size
USE_CACHE = True   # Whether to use cache
USE_PARALLEL = True  # Whether to use parallel processing


# =============================================================================
# Cache Decorators
# =============================================================================

@lru_cache(maxsize=128)
def cached_load_molecular_data(pdb_file: str) -> Tuple[md.Trajectory, np.ndarray, int]:
    """Cached version of molecular data loading"""
    return load_molecular_data(pdb_file)


@lru_cache(maxsize=128)
def cached_get_atom_types(pdb_file: str, ff_xml: str) -> np.ndarray:
    """Cached version of atom type extraction"""
    return get_atom_types(pdb_file, ff_xml)


# =============================================================================
# Utility Functions
# =============================================================================

@jit
def calculate_center_of_mass(positions: jnp.ndarray, masses: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate center of mass position
    
    Args:
        positions: Atomic positions [n_atoms, 3]
        masses: Atomic masses [n_atoms]
    
    Returns:
        jnp.ndarray: Center of mass position [3]
    """
    total_mass = jnp.sum(masses)
    center_of_mass = jnp.sum(positions * masses[:, None], axis=0) / total_mass
    return center_of_mass


@jit
def calculate_distance_between_centers_of_mass(
    pos1: jnp.ndarray, masses1: jnp.ndarray, 
    pos2: jnp.ndarray, masses2: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate the distance between the centers of mass of two molecules
    
    Args:
        pos1: Atomic positions of molecule 1 [n_atoms1, 3]
        masses1: Atomic masses of molecule 1 [n_atoms1]
        pos2: Atomic positions of molecule 2 [n_atoms2, 3]
        masses2: Atomic masses of molecule 2 [n_atoms2]
    
    Returns:
        jnp.ndarray: Distance between centers of mass
    """
    com1 = calculate_center_of_mass(pos1, masses1)
    com2 = calculate_center_of_mass(pos2, masses2)
    distance = jnp.linalg.norm(com2 - com1)
    return distance


@jit
def calculate_distance_between_centers_of_mass_vmap(
    pos1: jnp.ndarray, masses1: jnp.ndarray, 
    pos2: jnp.ndarray, masses2: jnp.ndarray
) -> jnp.ndarray:
    """
    Vectorized calculation of distances between centers of mass for multiple pairs of molecules
    
    Args:
        pos1: Batch of atomic positions for molecule 1 [n_batch, n_atoms1, 3]
        masses1: Atomic masses of molecule 1 [n_atoms1]
        pos2: Batch of atomic positions for molecule 2 [n_batch, n_atoms2, 3]
        masses2: Atomic masses of molecule 2 [n_atoms2]
    
    Returns:
        jnp.ndarray: Batch of distances between centers of mass [n_batch]
    """
    return vmap(calculate_distance_between_centers_of_mass, in_axes=(0, None, 0, None))(
        pos1, masses1, pos2, masses2
    )


def get_all_homo_key(data: Dict[str, Any], arr: List[str]) -> List[str]:
    """
    Get all keys for the same type of molecule
    
    Args:
        data: Data dictionary
        arr: List of molecule types
    
    Returns:
        List[str]: List of keys for the same type of molecule
    """
    dimer_test = []
    for key in data:
        a, b = key.split('_')[-2:]
        if a == b and b in arr:
            dimer_test.append(key)
    return dimer_test

def get_all_contain_key(data: Dict[str, Any], arr: List[str]) -> List[str]:
    """
    Get all keys for molecule pairs where both monomers are in the specified list
    
    Args:
        data: Data dictionary
        arr: List of molecule types
    
    Returns:
        List[str]: List of keys for molecule pairs
    """
    dimer_test = []
    for key in data:
        a, b = key.split('_')[-2:]
        if a in arr and b in arr:
            dimer_test.append(key)
    return dimer_test


def get_one_contain_key(data: Dict[str, Any], arr: List[str]) -> List[str]:
    """
    Get all keys for molecule pairs where at least one monomer is in the specified list
    
    Args:
        data: Data dictionary
        arr: List of molecule types
    
    Returns:
        List[str]: List of keys for molecule pairs
    """
    dimer_test = []
    for key in data:
        a, b = key.split('_')[-2:]
        if a in arr or b in arr:
            dimer_test.append(key)
    return dimer_test

def load_molecular_data(pdb_file: str) -> Tuple[md.Trajectory, np.ndarray, int]:
    """
    Load molecular data
    
    Args:
        pdb_file: Path to PDB file
    
    Returns:
        Tuple: (Trajectory object, atom masses array, number of atoms)
    """
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"File does not exist: {pdb_file}")
    
    traj = md.load(pdb_file)
    masses = traj.topology.to_dataframe()[0]['element'].apply(
        md.element.Element.getBySymbol
    ).apply(lambda e: e.mass).values
    n_atoms = traj.topology.n_atoms
    return traj, masses, n_atoms


def get_atom_types(pdb_file: str, ff_xml: str) -> np.ndarray:
    """
    Get atom type indices
    
    Args:
        pdb_file: Path to PDB file
        ff_xml: Path to force field file
    
    Returns:
        np.ndarray: Array of atom type indices
    """
    mol = PDBFile(pdb_file)
    H = Hamiltonian(ff_xml)
    pots = H.createPotential(
        mol.topology, 
        nonbondedCutoff=8*angstrom, 
        nonbondedMethod=CutoffPeriodic, 
        ethresh=1e-4
    )
    return pots.meta['ADMPPmeForce_map_atomtype']


def write_xyz_frame(
    file, num_atoms: int, elements: List[str], 
    posA: np.ndarray, posB: np.ndarray, 
    atype_indices_A: np.ndarray, atype_indices_B: np.ndarray,
    Natom_A: int, Natom_B: int, 
    target_energy: float, target_sr_energy: float, 
    distance: float, Comp: str, Nmols: int = 2
) -> None:
    """
    Write one frame of data in XYZ format
    
    Args:
        file: File object
        num_atoms: Total number of atoms
        elements: List of element symbols
        posA: Positions of molecule A
        posB: Positions of molecule B
        atype_indices_A: Atom type indices of molecule A
        atype_indices_B: Atom type indices of molecule B
        Natom_A: Number of atoms in molecule A
        Natom_B: Number of atoms in molecule B
        target_energy: Target energy
        target_sr_energy: Short-range energy
        distance: Distance between centers of mass
        Comp: Component information
        Nmols: Number of molecules
    """
    # Write number of atoms
    file.write(f"{num_atoms}\n")
    
    # Write comment line
    comment_line = (f'Lattice="30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0" '
                   f'femat.gzmat=T Properties=species:S:1:pos:R:3:molID:I:1:atype:I:1:forces:R:3 '
                   f'Nmols={Nmols} Comp={Comp} energy={target_energy:.6f} '
                   f'sr_energy={target_sr_energy:.6f} pbc="F F F" distance={distance:.6f}\n')
    file.write(comment_line)
    
    # Batch write atoms of molecule A
    for i_atom in range(Natom_A):
        pos = posA[i_atom]
        pos_str = "       ".join(f"{x:.8f}" for x in pos)
        file.write(f"{elements[i_atom]}   {pos_str}  0 {atype_indices_A[i_atom]} {pos_str}\n")
    
    # Batch write atoms of molecule B
    for j_atom in range(Natom_B):
        pos = posB[j_atom]
        pos_str = "       ".join(f"{x:.8f}" for x in pos)
        file.write(f"{elements[j_atom + Natom_A]}   {pos_str}  1 {atype_indices_B[j_atom]} {pos_str}\n")


def process_single_key(key: str, data: Dict[str, Any], ff_xml: str) -> List[str]:
    """
    Process data for a single key
    
    Args:
        key: Data key
        data: Data dictionary
        ff_xml: Path to force field file
    
    Returns:
        List[str]: List of generated XYZ format strings
    """
    results = []
    
    try:
        # Parse key values
        conf, numb_conf, monomer_A, monomer_B = key.split('_')
        
        # File paths
        dimer_file = f'dimer_bank/dimer_{numb_conf}_{monomer_A}_{monomer_B}.pdb'
        pdb_A_file = f'pdb_bank/{monomer_A}.pdb'
        pdb_B_file = f'pdb_bank/{monomer_B}.pdb'
        
        # Check if files exist
        if not all(os.path.exists(f) for f in [dimer_file, pdb_A_file, pdb_B_file]):
            print(f"Warning: File does not exist, skipping {key}")
            return results
        
        # Load molecular data (cached)
        if USE_CACHE:
            traj_AB = md.load(dimer_file)
            traj_A, pdb_A_mass, Natom_A = cached_load_molecular_data(pdb_A_file)
            traj_B, pdb_B_mass, Natom_B = cached_load_molecular_data(pdb_B_file)
            atype_indices_A = cached_get_atom_types(pdb_A_file, ff_xml)
            atype_indices_B = cached_get_atom_types(pdb_B_file, ff_xml)
        else:
            traj_AB = md.load(dimer_file)
            traj_A, pdb_A_mass, Natom_A = load_molecular_data(pdb_A_file)
            traj_B, pdb_B_mass, Natom_B = load_molecular_data(pdb_B_file)
            atype_indices_A = get_atom_types(pdb_A_file, ff_xml)
            atype_indices_B = get_atom_types(pdb_B_file, ff_xml)
        
        # Get element info
        elements = [atom.element.symbol for atom in traj_AB.topology.atoms]
        num_atoms = len(elements)
        Comp = f'{monomer_A}(1):{monomer_B}(1)'
        
        # Get scan results
        scan_res = data[key]
        posA_all = scan_res['posA']
        posB_all = scan_res['posB']
        target_energies = scan_res['tot_full'] - scan_res['ff_tot']
        target_sr_energies = scan_res['ff_tot']
        target_tot_full_energies = scan_res['tot_full']
        
        # Calculate distances between centers of mass (vectorized)
        distances = calculate_distance_between_centers_of_mass_vmap(
            posA_all, pdb_A_mass, posB_all, pdb_B_mass
        )
        
        # Process each frame
        for i in range(len(scan_res['tot'])):
            # Get current frame data
            posA = posA_all[i]
            posB = posB_all[i]
            target_energy = target_energies[i]
            target_sr_energy = target_sr_energies[i]
            target_tot_full_energy = target_tot_full_energies[i]  # Total energy
            distance = distances[i]
            
            # ===== Outlier filtering logic (includes total energy deviation judgment) =====
            # 1. Check for NaN or infinity (absolute outliers)
            # if (np.isnan(target_energy) or np.isinf(target_energy) or
            #     np.isnan(distance) or np.isinf(distance) or
            #     np.isnan(target_tot_full_energy) or np.isinf(target_tot_full_energy)):  # Added NaN/Inf check for total energy
            #     print(f"Filter outlier (NaN/Inf): Frame {i}, total energy={target_tot_full_energy}, energy={target_energy}, distance={distance}")
            #     continue
            
            # 2. Check if total energy (target_tot_full_energy) is within a reasonable range (Core: energy deviation filtering)
            if not (TOT_FULL_ENERGY_LOWER <= target_tot_full_energy <= TOT_FULL_ENERGY_UPPER):
                print(f"Filter total energy outlier: Frame {i}, total energy={target_tot_full_energy} (exceeds range [{TOT_FULL_ENERGY_LOWER}, {TOT_FULL_ENERGY_UPPER}])")
                continue
            
            # 3. Check other energy indicators (original logic retained)
            if not (ENERGY_LOWER <= target_energy <= ENERGY_UPPER):
                print(f"Filter energy outlier: Frame {i}, energy={target_energy} (exceeds range [{ENERGY_LOWER}, {ENERGY_UPPER}])")
                continue
            
            # # 4. Check if distance is within reasonable range (original logic retained)
            # if not (DISTANCE_LOWER <= distance <= DISTANCE_UPPER):
            #     print(f"Filter distance outlier: Frame {i}, distance={distance} (exceeds range [{DISTANCE_LOWER}, {DISTANCE_UPPER}])")
            #     continue
        
            # Build XYZ format string
            xyz_content = []
            xyz_content.append(f"{num_atoms}")
            xyz_content.append(
                f'Lattice="30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0" '
                f'femat.gzmat=T Properties=species:S:1:pos:R:3:molID:I:1:atype:I:1:forces:R:3 '
                f'Nmols=2 Comp={Comp} energy={target_energy:.6f} '
                f'sr_energy={target_sr_energy:.6f} pbc="F F F" distance={distance:.6f}'
            )
            
            # Add atoms of molecule A
            for i_atom in range(Natom_A):
                pos = posA[i_atom]
                pos_str = "       ".join(f"{x:.8f}" for x in pos)
                xyz_content.append(f"{elements[i_atom]}   {pos_str}  0 {atype_indices_A[i_atom]} {pos_str}")
            
            # Add atoms of molecule B
            for j_atom in range(Natom_B):
                pos = posB[j_atom]
                pos_str = "       ".join(f"{x:.8f}" for x in pos)
                xyz_content.append(f"{elements[j_atom + Natom_A]}   {pos_str}  1 {atype_indices_B[j_atom]} {pos_str}")
            
            results.append("\n".join(xyz_content))
    
    except Exception as e:
        print(f"Error processing {key}: {e}")
    
    return results


# =============================================================================
# Main Program
# =============================================================================

def main():
    """Main function"""
    print("Starting to load data...")
    
    # Load data
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} data entries")
    
    # Get molecule data
    dimer_test = get_all_homo_key(data, ['DME', 'EC', 'DMC','Li','FSI'])
    dimer_test = get_all_contain_key(data, ['DMC','Li','FSI'])
    dimer_test = get_all_contain_key(data, ['DME', 'EC', 'DMC'])
    arr = ['DEC', 'DFEA', 'DFEC', 'DMC', 'DME', 'DOL', 'EC', 'EMC', 'EP', \
            'FEC', 'FEMC', 'GBL', 'PC', 'PP', 'PS', 'SL', 'Li', 'FSI'] 
    dimer_test = get_all_contain_key(data, arr)  
    dimer_test = get_all_homo_key(data, ['DME'])
    dimer_test = get_one_contain_key(data, ['Li','Na'])

    dimer_test = ['conf_060_Li_DMC', 'conf_062_Li_EC', 'conf_051_Li_PF6']
    # dimer_test = list(data.keys())
    
    print(f"Found {len(dimer_test)} molecule pairs")
    
    # Statistics variables
    num_pairs = 0
    total_frames = 0
    
    # Batch process data
    print("Starting molecule data processing...")
    
    if USE_PARALLEL and len(dimer_test) > 1:
        # Parallel processing
        print("Using parallel processing...")
        all_results = []
        
        with ThreadPoolExecutor(max_workers=min(mp.cpu_count(), 8)) as executor:
            # Submit all tasks
            future_to_key = {
                executor.submit(process_single_key, key, data, FF_XML): key 
                for key in dimer_test
            }
            
            # Collect results
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    num_pairs += len(results)
                    print(f"Finished processing: {key} (generated {len(results)} frames)")
                except Exception as e:
                    print(f"Error processing {key}: {e}")
        
        # Write to file
        print("Writing to file...")
        with open(OUTFILE, 'w') as file:
            for result in all_results:
                file.write(result + "\n")
    
    else:
        # Serial processing
        print("Using serial processing...")
        with open(OUTFILE, 'w') as file:
            for key in dimer_test:
                print(f"Processing: {key}")
                
                # Parse key values
                conf, numb_conf, monomer_A, monomer_B = key.split('_')
                
                # File paths
                dimer_file = f'dimer_bank/dimer_{numb_conf}_{monomer_A}_{monomer_B}.pdb'
                pdb_A_file = f'pdb_bank/{monomer_A}.pdb'
                pdb_B_file = f'pdb_bank/{monomer_B}.pdb'
                
                # Check if files exist
                if not all(os.path.exists(f) for f in [dimer_file, pdb_A_file, pdb_B_file]):
                    print(f"Warning: File does not exist, skipping {key}")
                    continue
                
                # Load molecular data
                if USE_CACHE:
                    traj_AB = md.load(dimer_file)
                    traj_A, pdb_A_mass, Natom_A = cached_load_molecular_data(pdb_A_file)
                    traj_B, pdb_B_mass, Natom_B = cached_load_molecular_data(pdb_B_file)
                    atype_indices_A = cached_get_atom_types(pdb_A_file, FF_XML)
                    atype_indices_B = cached_get_atom_types(pdb_B_file, FF_XML)
                else:
                    traj_AB = md.load(dimer_file)
                    traj_A, pdb_A_mass, Natom_A = load_molecular_data(pdb_A_file)
                    traj_B, pdb_B_mass, Natom_B = load_molecular_data(pdb_B_file)
                    atype_indices_A = get_atom_types(pdb_A_file, FF_XML)
                    atype_indices_B = get_atom_types(pdb_B_file, FF_XML)
                
                # Get element information
                elements = [atom.element.symbol for atom in traj_AB.topology.atoms]
                num_atoms = len(elements)
                Comp = f'{monomer_A}(1):{monomer_B}(1)'
                
                # Get scan results
                scan_res = data[key]
                posA_all = scan_res['posA']
                posB_all = scan_res['posB']
                target_energies = scan_res['tot']
                #target_energies = scan_res['tot_full'] - scan_res['ff_tot']
                target_sr_energies = scan_res['ff_tot']
                target_tot_full_energies = scan_res['tot_full']
                
                # Calculate distances between centers of mass (vectorized)
                distances = calculate_distance_between_centers_of_mass_vmap(
                    posA_all, pdb_A_mass, posB_all, pdb_B_mass
                )
                
                # Process each frame
                for i in range(len(scan_res['tot'])):
                    total_frames += 1
                    # Get current frame data
                    posA = posA_all[i]
                    posB = posB_all[i]
                    target_energy = target_energies[i]
                    target_sr_energy = target_sr_energies[i]
                    target_tot_full_energy = target_tot_full_energies[i]  # Total energy
                    distance = distances[i]
                    
                    # ===== Outlier filtering logic (includes total energy deviation judgment) =====
                    # 1. Check for NaN or infinity (absolute outliers)
                    # if (np.isnan(target_energy) or np.isinf(target_energy) or
                    #     np.isnan(distance) or np.isinf(distance) or
                    #     np.isnan(target_tot_full_energy) or np.isinf(target_tot_full_energy)):  # Added NaN/Inf check for total energy
                    #     print(f"Filter outlier (NaN/Inf): Frame {i}, total energy={target_tot_full_energy}, energy={target_energy}, distance={distance}")
                    #     continue
                    
                    # 2. Check if total energy (target_tot_full_energy) is within a reasonable range (Core: energy deviation filtering)
                    if not (TOT_FULL_ENERGY_LOWER <= target_tot_full_energy <= TOT_FULL_ENERGY_UPPER):
                        print(f"Filter total energy outlier: Frame {i}, total energy={target_tot_full_energy} (exceeds range [{TOT_FULL_ENERGY_LOWER}, {TOT_FULL_ENERGY_UPPER}])")
                        continue
                    
                    # 3. Check other energy indicators (original logic retained)
                    if not (ENERGY_LOWER <= target_energy <= ENERGY_UPPER):
                        print(f"Filter energy outlier: Frame {i}, energy={target_energy} (exceeds range [{ENERGY_LOWER}, {ENERGY_UPPER}])")
                        continue
                    
                    # # 4. Check if distance is within reasonable range (original logic retained)
                    # if not (DISTANCE_LOWER <= distance <= DISTANCE_UPPER):
                    #     print(f"Filter distance outlier: Frame {i}, distance={distance} (exceeds range [{DISTANCE_LOWER}, {DISTANCE_UPPER}])")
                    #     continue
                    write_xyz_frame(
                        file, num_atoms, elements, posA, posB,
                        atype_indices_A, atype_indices_B, Natom_A, Natom_B,
                        target_energy, target_sr_energy, distance, Comp
                    )
                    num_pairs += 1
    
    print(f"Processing complete!")
    print(f"Total frames: {total_frames}")
    print(f"Number of valid pairs: {num_pairs}")
    print(f"Output file: {OUTFILE}")


if __name__ == "__main__":
    main()
