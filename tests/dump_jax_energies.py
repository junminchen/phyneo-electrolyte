import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import pickle
import numpy as np

import jax
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phyneo.models.eapnn import EAPNNForce

def get_dummy_inputs():
    n_atoms = 10
    n_atype = 10
    np.random.seed(42)
    pos = np.random.rand(n_atoms, 3) * 5.0
    box = np.eye(3) * 20.0
    pairs = np.array([[0, 3], [1, 5], [2, 7], [4, 8], [6, 9]])
    valid_mask = np.ones(5)
    topo_nblist = np.array([
        [1, -1, -1], [0, 2, -1], [1, -1, -1], [4, -1, -1], [3, -1, -1],
        [6, -1, -1], [5, -1, -1], [8, -1, -1], [7, -1, -1], [-1, -1, -1]
    ])
    topo_mask = (topo_nblist != -1)
    mol_ID = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4])
    atype_indices = np.random.randint(0, 10, size=n_atoms)
    return pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices

def main():
    jax_model = EAPNNForce(
        n_atoms=10, n_atype=10, rc=6.0, acsf_nmu=20, apsf_nmu=20, acsf_eta=100., apsf_eta=50.
    )
    
    pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices = get_dummy_inputs()
    
    key = jax.random.PRNGKey(0)
    # the signature of init: self.init(rngs, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
    params = jax_model.init(
        key, jnp.array(pos), jnp.array(box), jnp.array(pairs), jnp.array(valid_mask),
        jnp.array(topo_nblist), jnp.array(topo_mask), jnp.array(mol_ID), jnp.array(atype_indices)
    )
    jax_params = params['params'] if 'params' in params else params
    
    with open("tests/random_jax_params.pickle", "wb") as f:
        pickle.dump(jax_params, f)
    
    energy_jax = jax_model.apply(
        {'params': jax_params},
        jnp.array(pos), jnp.array(box), jnp.array(pairs), jnp.array(valid_mask),
        jnp.array(topo_nblist), jnp.array(topo_mask), jnp.array(mol_ID), jnp.array(atype_indices)
    )
    
    np.savez("tests/jax_outputs.npz", energy=np.array(energy_jax), pos=pos, box=box, pairs=pairs,
             valid_mask=valid_mask, topo_nblist=topo_nblist, topo_mask=topo_mask,
             mol_ID=mol_ID, atype_indices=atype_indices)
    print("JAX outputs saved successfully.")

if __name__ == "__main__":
    main()
