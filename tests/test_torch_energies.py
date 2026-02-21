import os
import sys
import pickle
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phyneo.models.torch_models import EAPNNForceTorch

def convert_params_to_torch(jax_params, torch_model):
    state_dict = torch_model.state_dict()
    print("Available keys in jax_params:", jax_params.keys())
    nn_params = jax_params.get('NeuralNetwork_0', jax_params.get('neural_network'))
    if nn_params is None:
        print("Neural network parameters not found! Returning.")
        return torch_model
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
                w = np.array(layer_dict['kernel']).T
                state_dict[f"{torch_prefix}.weight"] = torch.tensor(w, dtype=torch.float32)
            if 'bias' in layer_dict:
                b = np.array(layer_dict['bias'])
                state_dict[f"{torch_prefix}.bias"] = torch.tensor(b, dtype=torch.float32)
            if 'scale' in layer_dict:
                w = np.array(layer_dict['scale'])
                state_dict[f"{torch_prefix}.weight"] = torch.tensor(w, dtype=torch.float32)
                
    torch_model.load_state_dict(state_dict)
    return torch_model

def main():
    if not os.path.exists("tests/jax_outputs.npz"):
        print("Run dump_jax_energies.py first.")
        return
        
    data = np.load("tests/jax_outputs.npz")
    energy_jax = float(data['energy'])
    pos = data['pos']
    box = data['box']
    pairs = data['pairs']
    valid_mask = data['valid_mask']
    topo_nblist = data['topo_nblist']
    topo_mask = data['topo_mask']
    mol_ID = data['mol_ID']
    atype_indices = data['atype_indices']
    
    with open("tests/random_jax_params.pickle", 'rb') as f:
        jax_params = pickle.load(f)
    if 'params' in jax_params:
        jax_params = jax_params['params']
        
    zindex = [1, 3, 5, 6, 7, 8, 9, 11, 15, 16]
    
    torch_model = EAPNNForceTorch(
        n_atoms=10, n_atype=10, rc=6.0, zindex=zindex,
        acsf_nmu=20, apsf_nmu=20, acsf_eta=100., apsf_eta=50.
    )
    
    torch_model = convert_params_to_torch(jax_params, torch_model)
    torch_model.eval()
    
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
    
    np.testing.assert_allclose(energy_jax, energy_torch.item(), rtol=1e-4)
    print("Test passed! Energies match.")

if __name__ == "__main__":
    main()
