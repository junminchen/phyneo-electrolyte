import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import pickle
import numpy as np

import jax
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmff.sgnn.gnn import MolGNNForce
from dmff.sgnn.graph import TopGraph

def main():
    from dmff.sgnn.graph import from_pdb
    
    pdb_path = "data/pdb_bank/DMC.pdb"
    G = from_pdb(pdb_path)
        
    model = MolGNNForce(G, n_layers=(3, 2), sizes=[(40, 20, 20), (20, 10)], nn=1, sigma=1.0, mu=0.0)
    
    pos = G.positions
    box = np.eye(3) * 10.0
    G.set_box(box)
    
    energy_jax = model.forward(jnp.array(pos), jnp.array(box), model.params)
    
    np.savez("tests/jax_sgnn_outputs.npz", energy=np.array(energy_jax), pos=pos, box=box)
    
    params_to_save = {}
    for k, v in model.params.items():
        if isinstance(v, list):
            params_to_save[k] = [np.array(x) for x in v]
        else:
            params_to_save[k] = np.array(v)
            
    with open("tests/random_jax_sgnn_params.pickle", "wb") as f:
        pickle.dump(params_to_save, f)
        
    # extract G properties manually to avoid pickle closure errors
    topgraph_data = {
        'max_valence': getattr(G, 'max_valence', 4),
        'bonds': np.array(G.bonds),
        'b0': np.array(G.b0),
        'fscale_bond': float(G.fscale_bond),
        'angles': np.array(G.angles),
        'cos_a0': np.array(G.cos_a0),
        'fscale_angle': float(G.fscale_angle),
        'diheds': np.array(G.diheds),
        'feature_atypes': np.array(G.feature_atypes),
        'feature_indices': {k: np.array(v) for k, v in G.feature_indices.items()},
        'nb_connect': np.array(G.nb_connect) if G.nn == 1 else None,
        'weights': np.array(G.weights),
        'n_features': int(G.n_features),
    }
        
    with open("tests/dummy_topgraph.pickle", "wb") as f:
        pickle.dump(topgraph_data, f)
        
    print("JAX sGNN outputs saved successfully.")

if __name__ == "__main__":
    main()
