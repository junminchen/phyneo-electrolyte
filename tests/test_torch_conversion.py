import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import pickle
import numpy as np

import sys
try:
    if np.__version__.startswith("1."):
        sys.modules['numpy._core'] = np.core
except AttributeError:
    pass

# PyTorch
import torch

# JAX / Flax
import jax
import jax.numpy as jnp

# Local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phyneo.models.eapnn import EAPNNForce
from phyneo.models.torch_models import EAPNNForceTorch

def load_jax_params(pickle_path):
    with open(pickle_path, 'rb') as f:
        params = pickle.load(f)
    return params

def convert_params_to_torch(jax_params, torch_model):
    """
    jax_params: dict, nested dict of parameters from Flax model
    torch_model: nn.Module
    """
    state_dict = torch_model.state_dict()
    
    # EAPNN specific conversion
    if 'params' in jax_params:
        jax_params = jax_params['params']
    print("Available keys in jax_params:", jax_params.keys())
    
    if 'NeuralNetwork_0' not in jax_params:
        print("NeuralNetwork_0 not found! Returning.")
        return torch_model
        
    nn_params = jax_params['NeuralNetwork_0']
    
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
                # JAX is (in, out), Torch is (out, in)
                w = np.array(layer_dict['kernel']).T
                state_dict[f"{torch_prefix}.weight"] = torch.tensor(w, dtype=torch.float32)
            if 'bias' in layer_dict:
                b = np.array(layer_dict['bias'])
                state_dict[f"{torch_prefix}.bias"] = torch.tensor(b, dtype=torch.float32)
            if 'scale' in layer_dict: # LayerNorm
                w = np.array(layer_dict['scale'])
                state_dict[f"{torch_prefix}.weight"] = torch.tensor(w, dtype=torch.float32)
                
    torch_model.load_state_dict(state_dict)
    return torch_model

def get_dummy_inputs():
    n_atoms = 10
    n_atype = 10
    
    pos = np.random.rand(n_atoms, 3) * 10.0
    box = np.eye(3) * 20.0
    
    # 5 pairs
    pairs = np.array([
        [0, 1], [1, 2], [3, 4], [5, 6], [7, 8]
    ])
    valid_mask = np.ones(5)
    
    topo_nblist = np.array([
        [1, -1, -1], [0, 2, -1], [1, -1, -1], [4, -1, -1], [3, -1, -1],
        [6, -1, -1], [5, -1, -1], [8, -1, -1], [7, -1, -1], [-1, -1, -1]
    ])
    topo_mask = (topo_nblist != -1)
    
    mol_ID = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4])
    atype_indices = np.random.randint(0, 10, size=n_atoms)
    
    return pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices

def test_equivalence():
    pickle_path = "examples/md_simulation/params_ml.pickle"
    if not os.path.exists(pickle_path):
        print(f"Skipping test, {pickle_path} not found.")
        return
        
    jax_params = load_jax_params(pickle_path)
    if 'params' in jax_params:
        jax_params = jax_params['params']
        
    zindex = [1, 3, 5, 6, 7, 8, 9, 11, 15, 16]
        
    jax_model = EAPNNForce(
        n_atoms=10, 
        n_atype=10, 
        rc=6.0,  
        acsf_nmu=20,
        apsf_nmu=20,
        acsf_eta=100,
        apsf_eta=50
    )
    
    torch_model = EAPNNForceTorch(
        n_atoms=10,
        n_atype=10,
        rc=6.0,
        zindex=zindex,
        acsf_nmu=20,
        apsf_nmu=20,
        acsf_eta=100.,
        apsf_eta=50.
    )
    
    torch_model = convert_params_to_torch(jax_params, torch_model)
    torch_model.eval()
    
    pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices = get_dummy_inputs()
    
    # JAX eval
    energy_jax = jax_model.apply(
        {'params': jax_params},
        jnp.array(pos),
        jnp.array(box),
        jnp.array(pairs),
        jnp.array(valid_mask),
        jnp.array(topo_nblist),
        jnp.array(topo_mask),
        jnp.array(mol_ID),
        jnp.array(atype_indices)
    )
    
    # Torch eval
    with torch.no_grad():
        energy_torch = torch_model(
            torch.tensor(pos, dtype=torch.float32),
            torch.tensor(box, dtype=torch.float32),
            torch.tensor(pairs, dtype=torch.long),
            torch.tensor(valid_mask, dtype=torch.float32),
            torch.tensor(topo_nblist, dtype=torch.long),
            torch.tensor(topo_mask, dtype=torch.bool),
            torch.tensor(mol_ID, dtype=torch.long),
            torch.tensor(atype_indices, dtype=torch.long)
        )
        
    print(f"JAX Energy: {energy_jax}")
    print(f"Torch Energy: {energy_torch.item()}")
    
    np.testing.assert_allclose(float(energy_jax), energy_torch.item(), rtol=1e-4)
    print("Test passed! Energies match.")

if __name__ == "__main__":
    test_equivalence()
