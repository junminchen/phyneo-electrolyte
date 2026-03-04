import os
import sys
import pickle
import numpy as np
import torch
import openmm as mm
from openmm import app, unit

# Ensure JAX platforms issue doesn't break
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp

# Add repository root to path
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from phyneo.models.eapnn import EAPNNForce
from phyneo.models.torch_models import EAPNNForceTorch
from phyneo.interfaces.openmm_ml import EAPNNForceWrapper, create_and_add_torch_force
from phyneo.utils.data_utils import get_topology_neighbors, filter_and_pad_pairs, zindex as zindex_list

def load_eapnn_params_to_torch(torch_model, jax_params):
    state_dict = torch_model.state_dict()
    nn_params = jax_params.get('NeuralNetwork_0', jax_params.get('neural_network'))
    
    mapping = {
        'Dense_0': 'neural_network.layers.0',
        'LayerNorm_0': 'neural_network.layers.1',
        'Dense_1': 'neural_network.layers.3',
        'LayerNorm_1': 'neural_network.layers.4',
        'Dense_2': 'neural_network.layers.6',
        'LayerNorm_2': 'neural_network.layers.7',
        'Dense_3': 'neural_network.out_layer'
    }
    
    for jax_layer, torch_prefix in mapping.items():
        if jax_layer in nn_params:
            layer_dict = nn_params[jax_layer]
            if 'kernel' in layer_dict:
                state_dict[f"{torch_prefix}.weight"] = torch.tensor(np.array(layer_dict['kernel']).T, dtype=torch.float32)
            if 'bias' in layer_dict:
                state_dict[f"{torch_prefix}.bias"] = torch.tensor(np.array(layer_dict['bias']), dtype=torch.float32)
            if 'scale' in layer_dict: # LayerNorm
                state_dict[f"{torch_prefix}.weight"] = torch.tensor(np.array(layer_dict['scale']), dtype=torch.float32)
                
    torch_model.load_state_dict(state_dict)

def compare_eapnn_energies():
    pdb_path = os.path.join(repo_dir, "examples/2_training_pairwise_ml_nb/dimer_062_Li_EC.pdb")
    params_path = os.path.join(repo_dir, "examples/2_training_pairwise_ml_nb/results/best_model_params.pickle")
    
    if not os.path.exists(pdb_path) or not os.path.exists(params_path):
        print("Required files not found. Please ensure you are running from the repo root.")
        return

    # 1. Prepare Data
    pdb = app.PDBFile(pdb_path)
    topology = pdb.topology
    positions = np.array(pdb.positions.value_in_unit(unit.nanometer))
    # Move Li closer
    positions[0][1] = 1.8 
    
    box = np.array(topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    n_atoms = topology.getNumAtoms()
    
    # Get atom types and molecule IDs
    zindex_dict = {float(num): i for i, num in enumerate(zindex_list)}
    atype_indices = np.array([zindex_dict.get(float(atom.element.atomic_number), -1) for atom in topology.atoms()])
    mol_ID = np.array([atom.residue.index for atom in topology.atoms()])
    
    topo_nblist, topo_mask = get_topology_neighbors(pdb_path, connectivity=4, max_neighbors=20)
    
    from scipy.spatial import KDTree
    tree = KDTree(positions)
    all_pairs = tree.query_pairs(r=0.6) 
    pairs_list = list(all_pairs)
    pairs = np.zeros((len(pairs_list), 3), dtype=np.int32)
    if len(pairs_list) > 0:
        pairs[:, :2] = np.array(pairs_list)
    
    target_atypes = np.array([1, 7])
    max_pairs = max(len(pairs) + 100, 100)
    valid_pairs, valid_mask = filter_and_pad_pairs(pairs, atype_indices, target_atypes, max_pairs=max_pairs)
    
    with open(params_path, 'rb') as f:
        jax_full_params = pickle.load(f)
    jax_params = jax_full_params.get('params', jax_full_params)

    # 2. RUN JAX
    print("Running JAX evaluation...")
    jax_model = EAPNNForce(
        n_atoms=n_atoms, n_atype=len(zindex_list), rc=6.0,
        acsf_nmu=20, apsf_nmu=20, acsf_eta=100, apsf_eta=50
    )
    pos_jax = jnp.array(positions * 10.0)
    box_jax = jnp.array(box * 10.0)
    energy_jax_kcal = jax_model.apply(
        {'params': jax_params},
        pos_jax, box_jax, 
        jnp.array(valid_pairs), jnp.array(valid_mask), 
        jnp.array(topo_nblist), jnp.array(topo_mask), 
        jnp.array(mol_ID), jnp.array(atype_indices)
    )
    energy_jax_kj = float(energy_jax_kcal) * 4.184
    
    # 3. RUN PyTorch (Direct)
    print("Running PyTorch direct evaluation...")
    torch_model = EAPNNForceTorch(
        n_atoms=n_atoms, n_atype=len(zindex_list), rc=6.0, zindex=zindex_list,
        acsf_nmu=20, apsf_nmu=20, acsf_eta=100, apsf_eta=50
    )
    load_eapnn_params_to_torch(torch_model, jax_params)
    torch_model.eval()
    
    with torch.no_grad():
        energy_torch_kcal = torch_model(
            torch.tensor(positions * 10.0, dtype=torch.float32),
            torch.tensor(box * 10.0, dtype=torch.float32),
            torch.tensor(np.array(valid_pairs), dtype=torch.long),
            torch.tensor(np.array(valid_mask), dtype=torch.bool),
            torch.tensor(np.array(topo_nblist), dtype=torch.long),
            torch.tensor(np.array(topo_mask), dtype=torch.bool),
            torch.tensor(np.array(mol_ID), dtype=torch.long),
            torch.tensor(np.array(atype_indices), dtype=torch.long)
        )
    energy_torch_kj = float(energy_torch_kcal) * 4.184
    
    # 4. RUN OpenMM
    print("Running OpenMM verification...")
    system = mm.System()
    for _ in topology.atoms(): system.addParticle(12.0 * unit.amu)
    system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    wrapped_model = EAPNNForceWrapper(
        torch_model, np.array(valid_pairs), np.array(valid_mask), 
        np.array(topo_nblist), np.array(topo_mask), 
        np.array(mol_ID), np.array(atype_indices)
    )
    system = create_and_add_torch_force(system, wrapped_model)
    context = mm.Context(system, mm.VerletIntegrator(1.0*unit.fs), mm.Platform.getPlatformByName("CPU"))
    context.setPositions(positions * unit.nanometer)
    energy_omm_kj = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kj/unit.mole)

    # 5. Output Comparison
    print(f"\n=============================================")
    print(f"EAPNN JAX vs PyTorch Comparison:")
    print(f"JAX Energy (kJ/mol):    {energy_jax_kj:.6f}")
    print(f"Torch Energy (kJ/mol):  {energy_torch_kj:.6f}")
    print(f"OpenMM Energy (kJ/mol): {energy_omm_kj:.6f}")
    print(f"Difference (JAX-Torch): {abs(energy_jax_kj - energy_torch_kj):.6f}")
    print(f"=============================================\n")

if __name__ == "__main__":
    compare_eapnn_energies()
