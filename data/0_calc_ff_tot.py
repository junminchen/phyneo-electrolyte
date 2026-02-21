import pickle
import numpy as np
import time 
import sys 
import os 
import numpy as np

import jax
import jax.numpy as jnp
from jax import value_and_grad, vmap, jit

from openmm.app import PDBFile
from openmm.unit import angstrom
from openmm.app import CutoffPeriodic
from functools import partial
import pickle

from dmff.api import Hamiltonian
from dmff.utils import jit_condition
from dmff.common import nblist

class BasePairs:
    def __init__(self, ff, pdb, pdb_A, pdb_B):
        pdb = PDBFile(pdb)
        pdb_A = PDBFile(pdb_A)
        pdb_B = PDBFile(pdb_B)
        self.H = Hamiltonian(ff)
        self.pots = self.H.createPotential(pdb.topology, nonbondedCutoff=25*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4, step_pol=20)
        self.pots_A = self.H.createPotential(pdb_A.topology, nonbondedCutoff=25*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4, step_pol=20)
        self.pots_B = self.H.createPotential(pdb_B.topology, nonbondedCutoff=25*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4, step_pol=20)
        self.params = self.H.getParameters()

        self.pos = jnp.array(pdb.positions._value) #* 10
        self.pos_A = jnp.array(pdb_A.positions._value) #* 10
        self.pos_B = jnp.array(pdb_B.positions._value) #* 10

        self.box = jnp.eye(3) * 6 
        self.rc = 2.5
        self.nblist = nblist.NeighborList(self.box, self.rc, self.pots.meta['cov_map'])
        self.nblist_A = nblist.NeighborList(self.box, self.rc, self.pots_A.meta['cov_map'])
        self.nblist_B = nblist.NeighborList(self.box, self.rc, self.pots_B.meta['cov_map'])
        self.nblist.allocate(self.pos)
        self.nblist_A.allocate(self.pos_A)
        self.nblist_B.allocate(self.pos_B)
        self.pairs = self.nblist.pairs
        self.pairs_A = self.nblist_A.pairs
        self.pairs_B = self.nblist_B.pairs
        self.pairs_AB = self.pairs[self.pairs[:, 0] < self.pairs[:, 1]]
        self.pairs_A = self.pairs_A[self.pairs_A[:, 0] < self.pairs_A[:, 1]]
        self.pairs_B = self.pairs_B[self.pairs_B[:, 0] < self.pairs_B[:, 1]]

        self.etotal = self.pots.getPotentialFunc()
        self.etotal_A = self.pots_A.getPotentialFunc()
        self.etotal_B = self.pots_B.getPotentialFunc()

    def cal_E(self, pos_A, pos_B):
        # get position array
        pos_A *= 0.1
        pos_B *= 0.1
        pos_AB = jnp.concatenate([pos_A, pos_B], axis=0)
        box = self.box
        E = self.etotal(pos_AB, box, self.pairs_AB, self.params)\
               - self.etotal_A(pos_A, box, self.pairs_A, self.params)\
               - self.etotal_B(pos_B, box, self.pairs_B, self.params)
        return E


def get_all_contain_key(data,arr):
    """
    Get all keys containing the specified molecule types
    
    Args:
        data: Data dictionary
        arr: List of molecule types
    
    Returns:
        List[str]: List of keys containing the molecule types
    """
    dimer_test = []
    for key in data:
        a, b = key.split('_')[-2:]
        if a in arr and b in arr:
            dimer_test.append(key)
    return dimer_test

def get_all_homo_key(data, arr):
    dimer_test = []
    for key in data:
        a, b = key.split('_')[-2:]
        if a == b and b in arr:
            dimer_test.append(key)
        else:
            continue
    return dimer_test

ff = 'phyneo_ecl.xml'

print(ff)
params = Hamiltonian(ff).getParameters()

# data_file = 'data_bulk_scan.pickle'
data_file = 'data_dimer.pickle'
with open(data_file, 'rb') as ifile:
    data_batch = pickle.load(ifile)

# dimer_train = list(data_batch.keys())
# dimer_train.sort()
dimer_train = ['conf_060_Li_DMC', 'conf_062_Li_EC', 'conf_051_Li_PF6']

target_keys = ['tot_full', 'posA', 'posB']
data = {}
for conf in dimer_train:
    data[conf] = {}
    for k in target_keys:
        merged_list = []
        for subbatch in data_batch[conf]:
            arr = data_batch[conf][subbatch].get(k, None)
            if arr is not None:
                if isinstance(arr, np.ndarray):
                    merged_list.append(arr)
                else:
                    merged_list.append(np.array(arr))
        if merged_list:
            data[conf][k] = np.concatenate(merged_list)



# Loop to create subclasses and add them to the global namespace
class_instances = {}
cal_energy = {}
for pair in dimer_train: 
    print(pair)
    conf, numb_conf, monomer_A, monomer_B = pair.split('_')
    dimer_file = f'dimer_{numb_conf}_{monomer_A}_{monomer_B}'
    dimer_file = f'dimer_bank/{dimer_file}.pdb'
    pdb_A_file = f'pdb_bank/{monomer_A}.pdb'
    pdb_B_file = f'pdb_bank/{monomer_B}.pdb'
    class_instances[pair] = BasePairs(ff, dimer_file, pdb_A_file, pdb_B_file)
for class_name, class_instance in class_instances.items():
    cal_energy[class_name] = jit(vmap(class_instance.cal_E, in_axes=(0, 0), out_axes=(0)))

for key in dimer_train:

    scan_res = data[key]
    if 'tot_full' not in scan_res.keys():
        scan_res['tot_full'] = scan_res['tot'].copy()
    else: 
        scan_res['tot'] = scan_res['tot_full'].copy()

    npts = len(scan_res['tot'])
    # Calculate energy values
    E_sr = cal_energy[key](scan_res['posA'], scan_res['posB'])
    print(key, E_sr[0], len(E_sr))
    scan_res['ff_tot'] = E_sr
    scan_res['tot'] = scan_res['tot_full'] - E_sr

    # Save the updated data to a pickle file
    data_label = data_file.split('.')[0]
    with open(f'data_dimer_wt_ff.pickle', 'wb') as ofile:
        pickle.dump(data, ofile)
