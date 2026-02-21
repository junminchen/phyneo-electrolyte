import os
import pickle
import numpy as np
import torch
import openmm as mm
from openmm import app, unit
import sys

# Ensure JAX platforms issue doesn't break
os.environ["JAX_PLATFORMS"] = "cpu"

# Add repository root to path
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

# 1. Import PhyNEO components
from phyneo.models.torch_models import sGNNForceTorch
from phyneo.interfaces.openmm_ml import sGNNForceWrapper, create_and_add_torch_force
from dmff.sgnn.graph import from_pdb
from dmff.sgnn.gnn import MolGNNForce

def load_sgnn_params(model, pickle_path):
    with open(pickle_path, 'rb') as f:
        jax_params = pickle.load(f)
    if 'params' in jax_params: jax_params = jax_params['params']
    
    state_dict = model.state_dict()
    if 'w' in jax_params:
        state_dict['w'] = torch.tensor(np.array(jax_params['w']), dtype=torch.float32).reshape(1)
    for module_name, js_name in [('fc0', 'fc0'), ('fc1', 'fc1')]:
        if f'{js_name}.weight' in jax_params:
            for i, (w, b) in enumerate(zip(jax_params[f'{js_name}.weight'], jax_params[f'{js_name}.bias'])):
                state_dict[f'{module_name}.{i}.weight'] = torch.tensor(np.array(w), dtype=torch.float32)
                state_dict[f'{module_name}.{i}.bias'] = torch.tensor(np.array(b), dtype=torch.float32)
    if 'fc_final.weight' in jax_params:
        state_dict['fc_final.weight'] = torch.tensor(np.array(jax_params['fc_final.weight']), dtype=torch.float32)
        state_dict['fc_final.bias'] = torch.tensor(np.array(jax_params['fc_final.bias']), dtype=torch.float32).reshape(1)
    model.load_state_dict(state_dict)

def calculate_sgnn_energy(pdb_path, params_path):
    # 2. Setup Topology Graph
    G = from_pdb(pdb_path)
    _ = MolGNNForce(G, nn=1) # Initialize graph internal properties
    
    # 3. Create PyTorch Model
    model = sGNNForceTorch(G, n_layers=(3, 2), sizes=[(40, 20, 20), (20, 10)])
    load_sgnn_params(model, params_path)
    model.eval()
    
    # 4. Interface with OpenMM System
    pdb = app.PDBFile(pdb_path)
    system = mm.System()
    for _ in pdb.topology.atoms(): system.addParticle(1.0 * unit.amu)
    
    # Add TorchForce
    wrapped_model = sGNNForceWrapper(model)
    system = create_and_add_torch_force(system, wrapped_model)
    
    # Evaluate
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CPU"))
    context.setPositions(pdb.positions)
    
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    return energy

if __name__ == "__main__":
    PDB = os.path.join(repo_dir, "data/pdb_bank/EC.pdb")
    PARAMS = os.path.join(repo_dir, "examples/md_simulation/params_sgnn.pickle")
    
    if os.path.exists(PDB) and os.path.exists(PARAMS):
        energy = calculate_sgnn_energy(PDB, PARAMS)
        print(f"PDB: {PDB}")
        print(f"sGNN Energy in OpenMM: {energy}")
    else:
        print(f"Files not found: \n{PDB}\n{PARAMS}")
