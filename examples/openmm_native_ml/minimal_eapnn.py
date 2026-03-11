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
from phyneo.models.torch_models import EAPNNForceTorch
from phyneo.interfaces.openmm_ml import EAPNNForceWrapper, create_and_add_torch_force
from phyneo.utils.data_utils import get_topology_neighbors, filter_and_pad_pairs, zindex as zindex_list


def resolve_eapnn_params_path() -> str:
    candidates = [
        os.path.join(
            repo_dir,
            "examples/2_training_pairwise_ml_nb/ref_papar_model/params_LiNaPairs_no_force/model_params_epoch_810.pickle",
        ),
        os.path.join(
            repo_dir,
            "examples/2_training_pairwise_ml_nb/ref_papar_model/params_LiNaPairs_no_force/model_params_epoch_280.pickle",
        ),
        os.path.join(repo_dir, "examples/2_training_pairwise_ml_nb/results/best_model_params_fixed.pickle"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]

def load_eapnn_params(torch_model, jax_pickle_path):
    with open(jax_pickle_path, 'rb') as f:
        jax_params = pickle.load(f)
    if 'params' in jax_params: jax_params = jax_params['params']
    
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

def calculate_eapnn_energy(pdb_path, params_path):
    # 2. Extract topology and atom info
    pdb = app.PDBFile(pdb_path)
    topology = pdb.topology
    positions = np.array(pdb.positions.value_in_unit(unit.nanometer))
    
    # Move Li (atom 0) closer to EC to get non-zero energy if distance > rc
    # Li is at Y ~ 10.0, EC O is at Y ~ 21.0 in this specific dimer PDB. 
    # Move Li to Y = 1.8 nm (18.0 A)
    positions[0][1] = 1.8 
    
    box = topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    n_atoms = topology.getNumAtoms()
    
    # Get atom types and molecule IDs
    zindex_dict = {float(num): i for i, num in enumerate(zindex_list)}
    atype_indices = np.array([zindex_dict.get(float(atom.element.atomic_number), -1) for atom in topology.atoms()])
    mol_ID = np.array([atom.residue.index for atom in topology.atoms()])
    
    # Get topology neighbors (bonds)
    topo_nblist, topo_mask = get_topology_neighbors(pdb_path, connectivity=4, max_neighbors=20)
    
    # 3. Create Neighbor List (Pairs)
    from scipy.spatial import KDTree
    tree = KDTree(positions)
    # EAPNN rc=6.0 Angstrom = 0.6 nm
    all_pairs = tree.query_pairs(r=0.6) 
    pairs_list = list(all_pairs)
    
    if len(pairs_list) > 0:
        pairs = np.zeros((len(pairs_list), 3), dtype=np.int32)
        pairs[:, :2] = np.array(pairs_list)
    else:
        pairs = np.zeros((0, 3), dtype=np.int32)
    
    # Filter and pad as expected by the model
    target_atypes = np.array([1, 7]) # Li and Na
    max_pairs = max(len(pairs) + 100, 100)
    valid_pairs, valid_mask = filter_and_pad_pairs(pairs, atype_indices, target_atypes, max_pairs=max_pairs)
    
    # 4. Create PyTorch Model
    n_atype = len(zindex_list)
    model = EAPNNForceTorch(
        n_atoms=n_atoms, n_atype=n_atype, rc=6.0, zindex=zindex_list,
        acsf_nmu=20, apsf_nmu=20, acsf_eta=100, apsf_eta=50
    )
    load_eapnn_params(model, params_path)
    model.eval()
    
    # 5. Interface with OpenMM System
    system = mm.System()
    for _ in topology.atoms(): system.addParticle(12.0 * unit.amu)
    system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    
    # Add TorchForce
    wrapped_model = EAPNNForceWrapper(
        model, 
        np.array(valid_pairs), np.array(valid_mask), 
        np.array(topo_nblist), np.array(topo_mask), 
        np.array(mol_ID), np.array(atype_indices)
    )
    system = create_and_add_torch_force(system, wrapped_model)
    
    # Evaluate
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CPU"))
    context.setPositions(positions * unit.nanometer)
    
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    return energy

if __name__ == "__main__":
    # Correct relative paths from repo root
    PDB = os.path.join(repo_dir, "examples/2_training_pairwise_ml_nb/dimer_062_Li_EC.pdb")
    PARAMS = resolve_eapnn_params_path()
    
    if os.path.exists(PDB) and os.path.exists(PARAMS):
        energy = calculate_eapnn_energy(PDB, PARAMS)
        print(f"PDB: {PDB}")
        print(f"EAPNN Energy in OpenMM: {energy}")
    else:
        print(f"Files not found: \n{PDB}\n{PARAMS}")
