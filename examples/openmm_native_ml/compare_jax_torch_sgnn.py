import os
import sys
import pickle
import numpy as np
import torch
import openmm as mm
from openmm import app, unit

# Ensure JAX platforms issue doesn't break
os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp

# Add repository root to path
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from phyneo.interfaces.openmm_ml import sGNNForceWrapper, create_and_add_torch_force
from phyneo.models.torch_models import sGNNForceTorch
from dmff.sgnn.graph import from_pdb
from dmff.sgnn.gnn import MolGNNForce

def load_sgnn_params(model, jax_params):
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

def compare_sgnn_energies():
    pdb_path = os.path.join(repo_dir, "data/pdb_bank/EC.pdb")
    params_path = os.path.join(repo_dir, "examples/md_simulation/params_sgnn.pickle")
    
    if not os.path.exists(pdb_path) or not os.path.exists(params_path):
        print("Required files not found.")
        return

    # 1. Setup
    G = from_pdb(pdb_path)
    # Initialize JAX model to prep graph
    _ = MolGNNForce(G, nn=1)
    
    with open(params_path, 'rb') as f:
        jax_params = pickle.load(f)
    jax_params = jax_params.get('params', jax_params)

    # 2. RUN JAX
    print("Running JAX evaluation...")
    jax_model = MolGNNForce(G, nn=1, sigma=162.13, mu=117.42)
    pdb_obj = app.PDBFile(pdb_path)
    pos_A = jnp.array(pdb_obj.positions.value_in_unit(unit.angstrom))
    box_A = jnp.eye(3) * 30.0 # Standard box from PDB metadata if available
    energy_jax_kcal = jax_model.forward(pos_A, box_A, jax_params)
    energy_jax_kj = float(energy_jax_kcal) * 4.184
    
    # 3. RUN PyTorch (Direct)
    print("Running PyTorch direct evaluation...")
    torch_model = sGNNForceTorch(G, n_layers=(3, 2), sizes=[(40, 20, 20), (20, 10)], sigma=162.13, mu=117.42)
    load_sgnn_params(torch_model, jax_params)
    torch_model.eval()
    with torch.no_grad():
        energy_torch_kcal = torch_model(
            torch.tensor(np.array(pos_A), dtype=torch.float32),
            torch.tensor(np.array(box_A), dtype=torch.float32)
        )
    energy_torch_kj = float(energy_torch_kcal) * 4.184
    
    # 4. RUN OpenMM
    print("Running OpenMM verification...")
    system = mm.System()
    for _ in pdb_obj.topology.atoms(): system.addParticle(1.0 * unit.amu)
    system.setDefaultPeriodicBoxVectors(3.0*unit.nm, 3.0*unit.nm, 3.0*unit.nm)
    wrapped_model = sGNNForceWrapper(torch_model)
    system = create_and_add_torch_force(system, wrapped_model)
    context = mm.Context(system, mm.VerletIntegrator(1.0*unit.fs), mm.Platform.getPlatformByName("CPU"))
    context.setPositions(pdb_obj.positions)
    energy_omm_kj = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kj/unit.mole)

    # 5. Output Comparison
    print(f"\n=============================================")
    print(f"sGNN JAX vs PyTorch Comparison:")
    print(f"JAX Energy (kJ/mol):    {energy_jax_kj:.6f}")
    print(f"Torch Energy (kJ/mol):  {energy_torch_kj:.6f}")
    print(f"OpenMM Energy (kJ/mol): {energy_omm_kj:.6f}")
    print(f"Difference (JAX-Torch): {abs(energy_jax_kj - energy_torch_kj):.6f}")
    print(f"=============================================\n")

if __name__ == "__main__":
    compare_sgnn_energies()
