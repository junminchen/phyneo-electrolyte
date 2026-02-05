"""
Example: Molecular Dynamics simulation of liquid electrolyte system.

This example demonstrates how to run MD simulations using PhyNEO-Electrolyte
force field for a simple liquid electrolyte system.
"""

import numpy as np
from ase import Atoms
from ase.io import write
import torch

# Import PhyNEO models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from phyneo.models import SlaterTypeFunction, PairwiseMLCorrection
from phyneo.utils.md_utils import MDSimulator


class PhyNEOCalculator:
    """
    Simple calculator using PhyNEO-Electrolyte models.
    
    This is a simplified calculator for demonstration purposes.
    A full implementation would integrate all model components.
    """
    
    def __init__(self, slater_model=None, pairwise_model=None):
        self.slater_model = slater_model
        self.pairwise_model = pairwise_model
        self.results = {}
    
    def get_potential_energy(self, atoms=None):
        """Compute potential energy."""
        if atoms is not None:
            self.atoms = atoms
        
        # Simple harmonic potential for demonstration
        positions = self.atoms.get_positions()
        n_atoms = len(positions)
        
        energy = 0.0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(positions[j] - positions[i])
                # Simple Lennard-Jones-like potential
                if dist < 5.0:
                    energy += 4.0 * ((1.0 / dist)**12 - (1.0 / dist)**6)
        
        return energy
    
    def get_forces(self, atoms=None):
        """Compute forces (numerical differentiation for demo)."""
        if atoms is not None:
            self.atoms = atoms
        
        forces = np.zeros((len(self.atoms), 3))
        delta = 0.001
        
        for i in range(len(self.atoms)):
            for j in range(3):
                positions = self.atoms.get_positions().copy()
                
                # Forward difference
                positions[i, j] += delta
                self.atoms.set_positions(positions)
                e_plus = self.get_potential_energy()
                
                # Backward difference
                positions[i, j] -= 2 * delta
                self.atoms.set_positions(positions)
                e_minus = self.get_potential_energy()
                
                # Central difference
                forces[i, j] = -(e_plus - e_minus) / (2 * delta)
                
                # Restore positions
                positions[i, j] += delta
                self.atoms.set_positions(positions)
        
        return forces


def create_electrolyte_system():
    """
    Create a simple liquid electrolyte system.
    
    Returns:
        ase.Atoms: Initial atomic structure
    """
    # Create a simple box with Li+ and PF6- ions in solvent
    # This is a simplified example
    
    n_li = 4  # Number of Li+ ions
    n_pf6 = 4  # Number of PF6- ions (P + 6F per ion)
    
    # Create positions in a box
    box_size = 15.0  # Angstrom
    positions = []
    symbols = []
    
    # Add Li+ ions
    for i in range(n_li):
        pos = np.random.rand(3) * box_size
        positions.append(pos)
        symbols.append('Li')
    
    # Add PF6- ions (simplified as P atoms for demo)
    for i in range(n_pf6):
        pos = np.random.rand(3) * box_size
        positions.append(pos)
        symbols.append('P')
    
    # Create Atoms object
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[box_size, box_size, box_size],
        pbc=[True, True, True]
    )
    
    return atoms


def run_md_example():
    """
    Run a simple MD simulation example.
    """
    print("=" * 60)
    print("PhyNEO-Electrolyte MD Simulation Example")
    print("=" * 60)
    
    # Create system
    print("\nCreating electrolyte system...")
    atoms = create_electrolyte_system()
    print(f"System: {len(atoms)} atoms")
    print(f"Cell: {atoms.get_cell()}")
    
    # Save initial structure
    write('initial_structure.xyz', atoms)
    print("Saved initial structure to initial_structure.xyz")
    
    # Initialize calculator
    print("\nInitializing PhyNEO calculator...")
    calculator = PhyNEOCalculator()
    
    # Setup MD simulator
    print("\nSetting up MD simulation...")
    simulator = MDSimulator(
        atoms=atoms,
        calculator=calculator,
        temperature=300.0,  # K
        timestep=1.0,  # fs
        friction=0.01
    )
    
    # Run simulation
    print("\nRunning MD simulation...")
    print("Temperature: 300 K")
    print("Timestep: 1.0 fs")
    print("Steps: 100")
    print()
    
    results = simulator.run(
        nsteps=100,
        trajectory_file='trajectory.traj',
        log_interval=10
    )
    
    # Save final structure
    final_atoms = simulator.get_trajectory()
    write('final_structure.xyz', final_atoms)
    print("\nSaved final structure to final_structure.xyz")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Simulation Statistics")
    print("=" * 60)
    energies = results['energies']
    temperatures = results['temperatures']
    
    print(f"Average potential energy: {np.mean(energies[:, 0]):.3f} eV")
    print(f"Average kinetic energy: {np.mean(energies[:, 1]):.3f} eV")
    print(f"Average total energy: {np.mean(energies[:, 2]):.3f} eV")
    print(f"Average temperature: {np.mean(temperatures):.1f} K")
    print(f"Temperature std: {np.std(temperatures):.1f} K")
    
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run example
    run_md_example()
