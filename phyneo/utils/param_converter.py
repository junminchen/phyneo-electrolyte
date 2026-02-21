import os
import sys
import pickle
import numpy as np
import torch

try:
    if np.__version__.startswith("1."):
        sys.modules['numpy._core'] = np.core
except AttributeError:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phyneo.models.torch_models import EAPNNForceTorch, sGNNForceTorch

def convert_eapnn(jax_pickle_path, pt_output_path, model_kwargs):
    with open(jax_pickle_path, 'rb') as f:
        jax_params = pickle.load(f)
    if 'params' in jax_params:
        jax_params = jax_params['params']
        
    torch_model = EAPNNForceTorch(**model_kwargs)
    state_dict = torch_model.state_dict()
    
    nn_params = jax_params.get('NeuralNetwork_0', jax_params.get('neural_network'))
    if nn_params is None:
        raise ValueError("Neural network params not found in EAPNN pickle!")
        
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
    torch.save(torch_model.state_dict(), pt_output_path)
    print(f"EAPNN parameters saved to {pt_output_path}")

def convert_sgnn(jax_pickle_path, pt_output_path, model_kwargs):
    with open(jax_pickle_path, 'rb') as f:
        jax_params = pickle.load(f)
        
    # extract raw python dict if it's jax object
    if 'params' in jax_params:
        jax_params = jax_params['params']
        
    torch_model = sGNNForceTorch(**model_kwargs)
    state_dict = torch_model.state_dict()
    
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
    torch.save(torch_model.state_dict(), pt_output_path)
    print(f"sGNN parameters saved to {pt_output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert JAX params to PyTorch")
    parser.add_argument("--model", type=str, choices=["eapnn", "sgnn"], required=True)
    parser.add_argument("--input", type=str, required=True, help="Input pickle file")
    parser.add_argument("--output", type=str, required=True, help="Output .pt file")
    # For a real pipeline, we would load hyperparameters (like G for sGNN) dynamically.
    # We will instantiate models in the OpenMM interface script directly instead, 
    # but for standalone conversion we just save state_dict which is topology independent, 
    # except sGNN needs a TopGraph just to instantiate! We will bypass that in production by
    # extracting script logic, but for now we write the interface logic.
