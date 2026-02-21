import os
import sys

# Ensure JAX platforms issue doesn't break importing
os.environ["JAX_PLATFORMS"] = "cpu"

import openmm as mm
from openmm import app
from openmm import unit
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phyneo.utils.param_converter import convert_eapnn, convert_sgnn
from phyneo.interfaces.openmm_ml import EAPNNForceWrapper, sGNNForceWrapper, create_and_add_torch_force
from phyneo.models.torch_models import EAPNNForceTorch, sGNNForceTorch
from dmff.sgnn.graph import from_pdb
import pickle

def main():
    print("Setting up OpenMM System for TorchForce E2E Test...")
    
    # 1. Load PDB and create basic system
    pdb_path = "examples/md_simulation/init.pdb"
    pdb = app.PDBFile(pdb_path)
    
    # Load basic force field to get masses. We just make an empty system 
    # and add particles since we handle energies entirely through ML.
    system = mm.System()
    
    # We apply PBC from the PDB 
    box_vectors = pdb.topology.getPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    
    for atom in pdb.topology.atoms():
        # Set all masses to Carbon mass for dummy test
        system.addParticle(12.0)
    
    print(f"Added {system.getNumParticles()} particles to the system.")
    
    # 2. Convert JAX params to Torch on the fly for test
    eapnn_pickle = "examples/md_simulation/params_ml.pickle"
    sgnn_pickle = "examples/md_simulation/params_sgnn.pickle"
    
    eapnn_pt = "examples/md_simulation/eapnn.pt"
    sgnn_pt = "examples/md_simulation/sgnn.pt"
    
    n_atoms = system.getNumParticles()
    # EAPNN uses atype 11 for electrolytes... Let's just mock kwargs
    eapnn_kwargs = {
        'n_atoms': n_atoms,
        'n_atype': 10,  
        'rc': 6.0,
        'zindex': [1, 3, 5, 6, 7, 8, 9, 11, 15, 16],
        'acsf_nmu': 20,
        'apsf_nmu': 10,
        'acsf_eta': 100,
        'apsf_eta': 25
    }
    
    # Create sGNN Graph from PDB
    print("Building sGNN Graph...")
    G = from_pdb(pdb_path)
    # the examples use PF6 or other complicated things, we just need to get the first valid graph
    # from_pdb returns dict for multiple molecules. Let's merge them into one TopGraph or just test!
    # Wait, the ML model handles the WHOLE system! In DMFF, client_dmff.py builds a unified MolGNNForce.
    print("Converting sGNN params...")
    # Just do a mock instantiation to get past it! Wait, we don't have unified TopGraph in this test script.
    
    print("Test ready to run actual integration code once topologies are integrated.")
    
if __name__ == "__main__":
    main()
