import os
import sys
import pickle
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phyneo.models.torch_models import sGNNForceTorch

def convert_params_to_torch_sgnn(jax_params, torch_model):
    state_dict = torch_model.state_dict()
    
    # In JAX sGNN: prm_transform_f2i changes names to 'fc0.weight', etc.
    # We will map standard JAX keys directly to Torch
    # Actually wait: JAX MolGNNForce has 'w', 'fc0.weight', 'fc0.bias', 'fc1.weight', 'fc1.bias', 'fc_final.weight', 'fc_final.bias'
    # where weight is a LIST of arrays for each layer.
    
    # We map them
    if 'w' in jax_params:
        state_dict['w'] = torch.tensor(np.array(jax_params['w']), dtype=torch.float32).reshape(1)
        
    for module_name, js_name in [('fc0', 'fc0'), ('fc1', 'fc1')]:
        if f'{js_name}.weight' in jax_params:
            weights = jax_params[f'{js_name}.weight']
            biases = jax_params[f'{js_name}.bias']
            for i in range(len(weights)):
                w = np.array(weights[i])
                b = np.array(biases[i])
                state_dict[f'{module_name}.{i}.weight'] = torch.tensor(w, dtype=torch.float32)
                state_dict[f'{module_name}.{i}.bias'] = torch.tensor(b, dtype=torch.float32)
                
    if 'fc_final.weight' in jax_params:
        w = np.array(jax_params['fc_final.weight'])
        b = np.array(jax_params['fc_final.bias'])
        state_dict['fc_final.weight'] = torch.tensor(w, dtype=torch.float32)
        state_dict['fc_final.bias'] = torch.tensor(b, dtype=torch.float32).reshape(1)
        
    torch_model.load_state_dict(state_dict)
    return torch_model

def main():
    if not os.path.exists("tests/random_jax_sgnn_params.pickle"):
        print("Run dump_jax_sgnn.py first.")
        return
        
    with open("tests/random_jax_sgnn_params.pickle", 'rb') as f:
        jax_params = pickle.load(f)
        
    with open("tests/dummy_topgraph.pickle", 'rb') as f:
        g_data = pickle.load(f)
        
    class MockGraph:
        def __init__(self, data):
            for k, v in data.items():
                setattr(self, k, v)
    
    G = MockGraph(g_data)
        
    torch_model = sGNNForceTorch(G, n_layers=(3, 2), sizes=[(40, 20, 20), (20, 10)], nn_hops=1, sigma=1.0, mu=0.0)
    torch_model = convert_params_to_torch_sgnn(jax_params, torch_model)
    torch_model.eval()
    
    data = np.load("tests/jax_sgnn_outputs.npz")
    pos = data['pos']
    box = data['box']
    energy_jax = float(data['energy'])
    
    with torch.no_grad():
        pos_t = torch.tensor(pos, dtype=torch.float32)
        box_t = torch.tensor(box, dtype=torch.float32)
        energy_torch = torch_model(pos_t, box_t)
        
    print(f"JAX sGNN Energy: {energy_jax}")
    print(f"Torch sGNN Energy: {energy_torch.item()}")
    
    np.testing.assert_allclose(energy_jax, energy_torch.item(), rtol=1e-4)
    print("Test passed! sGNN Energies match.")

if __name__ == "__main__":
    main()
