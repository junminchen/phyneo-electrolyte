"""
Molecular dynamics utilities for PhyNEO-Electrolyte.
"""

import numpy as np
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io import write


class MDSimulator:
    """
    Molecular dynamics simulator using PhyNEO-Electrolyte force field.
    
    Args:
        atoms (ase.Atoms): Initial atomic structure
        calculator: Calculator for forces and energies
        temperature (float): Target temperature in Kelvin
        timestep (float): MD timestep in fs
        friction (float): Friction coefficient for Langevin dynamics (optional)
    """
    
    def __init__(
        self,
        atoms,
        calculator,
        temperature=300.0,
        timestep=1.0,
        friction=0.01
    ):
        self.atoms = atoms
        self.atoms.set_calculator(calculator)
        self.temperature = temperature
        self.timestep = timestep
        self.friction = friction
        
        # Initialize velocities
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature)
        
        # Setup MD integrator
        if friction > 0:
            # Langevin dynamics for NVT ensemble
            self.dynamics = Langevin(
                self.atoms,
                timestep=timestep * units.fs,
                temperature_K=temperature,
                friction=friction
            )
        else:
            # Velocity Verlet for NVE ensemble
            self.dynamics = VelocityVerlet(
                self.atoms,
                timestep=timestep * units.fs
            )
    
    def run(self, nsteps, trajectory_file=None, log_interval=10):
        """
        Run molecular dynamics simulation.
        
        Args:
            nsteps (int): Number of MD steps
            trajectory_file (str): Output trajectory file (optional)
            log_interval (int): Interval for logging
        
        Returns:
            dict: Simulation statistics (energies, temperatures, etc.)
        """
        energies = []
        temperatures = []
        
        # Attach trajectory writer if specified
        if trajectory_file:
            from ase.io.trajectory import Trajectory
            traj = Trajectory(trajectory_file, 'w', self.atoms)
            self.dynamics.attach(traj.write, interval=log_interval)
        
        # Run simulation
        for step in range(nsteps):
            self.dynamics.run(1)
            
            if step % log_interval == 0:
                epot = self.atoms.get_potential_energy()
                ekin = self.atoms.get_kinetic_energy()
                etot = epot + ekin
                temp = ekin / (1.5 * units.kB * len(self.atoms))
                
                energies.append([epot, ekin, etot])
                temperatures.append(temp)
                
                print(f"Step {step}: E_pot={epot:.3f} eV, "
                      f"E_kin={ekin:.3f} eV, T={temp:.1f} K")
        
        return {
            'energies': np.array(energies),
            'temperatures': np.array(temperatures)
        }
    
    def get_trajectory(self):
        """Get current atomic configuration."""
        return self.atoms.copy()
    
    def set_temperature(self, temperature):
        """Set new target temperature."""
        self.temperature = temperature
        if hasattr(self.dynamics, 'set_temperature'):
            self.dynamics.set_temperature(temperature_K=temperature)


def analyze_trajectory(trajectory_file):
    """
    Analyze MD trajectory.
    
    Args:
        trajectory_file (str): Path to trajectory file
    
    Returns:
        dict: Analysis results (RDF, MSD, etc.)
    """
    from ase.io import read
    
    traj = read(trajectory_file, ':')
    
    # Compute radial distribution function (example)
    # This is a simplified version - full implementation would be more complex
    
    results = {
        'n_frames': len(traj),
        'mean_energy': None,  # Would compute from trajectory
        'mean_temperature': None,
    }
    
    return results


def save_snapshot(atoms, filename, format='xyz'):
    """
    Save atomic snapshot.
    
    Args:
        atoms (ase.Atoms): Atomic structure
        filename (str): Output filename
        format (str): Output format
    """
    write(filename, atoms, format=format)
