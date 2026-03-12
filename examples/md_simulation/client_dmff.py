#!/usr/bin/env python3
import os
import sys
import driver
import numpy as np
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *
import pickle

from dmff.api import Hamiltonian
from dmff.common import nblist
import jax
from jax import jit, value_and_grad, vmap
import jax.numpy as jnp

from eapnn import *
from phyneo.utils import (
    DEFAULT_ABN_RESIDUE_NAMES,
    build_sgnn_model_bundle,
    find_residue_blocks,
    group_residue_blocks_by_name,
    non_residue_atom_indices,
    resolve_default_sgnn_specs,
    spec_for_residue_name,
    stack_positions_for_blocks,
)

class DMFFDriver(driver.BaseDriver):

    def __init__(self, addr, port, socktype):
        #addr = addr + '_%s'%os.environ['SLURM_JOB_ID']
        # set up the interface with ipi
        driver.BaseDriver.__init__(self, port, addr, socktype)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        pdb = os.path.join(base_dir, 'init.pdb')
        ff_xml = os.path.join(base_dir, 'phyneo_ecl.xml')
        psr1 = os.path.join(base_dir, 'params_ml.pickle')
        sgnn_specs = resolve_default_sgnn_specs(base_dir)
        residue_names = DEFAULT_ABN_RESIDUE_NAMES

        mol = PDBFile(pdb) 
        pos = jnp.array(mol.positions._value) 
        box = jnp.array(mol.topology.getPeriodicBoxVectors()._value)
        L = box[0][0]
        
        rc = 0.6
        H = Hamiltonian(ff_xml)
        pots = H.createPotential(mol.topology, nonbondedCutoff=rc*nanometer, nonbondedMethod=PME, ethresh=1e-4, step_pol=10)
        efunc_nb = pots.getPotentialFunc()
        params_nb = H.getParameters()

        # neighbor list
        self.nbl = nblist.NeighborListFreud(box, rc, pots.meta['cov_map'])
        # self.nbl.capacity_multiplier = 500000 # avoid pairs leaking
        self.nbl.allocate(pos, box)
        self.pairs = self.nbl.pairs


        # define atomic symbols and corresponding indexes
        atom_elements = []
        for atom in mol.topology.atoms():
            atom_elements.append(atom.element.atomic_number)
        z_atomnum = jnp.array(atom_elements)

        zindex = [1, 3, 5, 6, 7, 8, 9, 11, 15, 16]
        n_atype = len(zindex)
        z_atomnum_list = [float(num) for num in np.array(z_atomnum)]
        zindex_dict = {float(num): i for i, num in enumerate(zindex)}
        self.atype_indices = jnp.array([zindex_dict.get(num, -1) for num in z_atomnum_list])

        mol_ID = []
        for atom in mol.topology.atoms():
            mol_ID.append(atom.residue.index)
        mol_ID = jnp.array(mol_ID)

        topo_nblist, topo_mask = get_topology_neighbors(pdb, connectivity=4, max_neighbors=20, max_n_atoms=None)

        n_atoms = len(pos)
        atomic_nums = jnp.array([atom.element.atomic_number for atom in mol.topology.atoms()], dtype=int)
        # 标记Li(3)和Na(11)原子
        target_mask = (atomic_nums == 3) | (atomic_nums == 11)
        target_indices = jnp.where(target_mask)[0]
        self.max_pairs = len(target_indices)*100

        self.valid_pairs, self.valid_mask = filter_and_pad_pairs(self.pairs, self.atype_indices, max_pairs=self.max_pairs)


        model_nb = EAPNNForce(
            n_atoms=n_atoms, 
            n_atype=n_atype, 
            rc=6.0,  
            acsf_nmu=20,
            apsf_nmu=20,
            acsf_eta=100,
            apsf_eta=50
        )

        key = jax.random.PRNGKey(0)
        model_nb.init(key, pos*10, box*10, self.valid_pairs, self.valid_mask, topo_nblist, topo_mask, mol_ID, self.atype_indices)

        with open(psr1, 'rb') as ifile:
            params = pickle.load(ifile)	

        abn_blocks = find_residue_blocks(mol.topology, residue_names)
        abn_blocks_by_name = group_residue_blocks_by_name(abn_blocks)
        non_abn_indices = non_residue_atom_indices(len(pos), abn_blocks)

        if abn_blocks:
            found_names = ", ".join(sorted({block.name for block in abn_blocks}))
            print(f"Topology contains ABn residues: {found_names}.")
        else:
            print("Topology does not contain ABn residue.")

        standard_bundle = None
        if non_abn_indices.size > 0:
            standard_pdb = os.path.join(
                base_dir,
                'init_remaining.pdb' if abn_blocks else 'init.pdb',
            )
            standard_bundle = build_sgnn_model_bundle(standard_pdb, sgnn_specs['standard'])

        abn_bundles = {}
        abn_batch_forward = {}
        for residue_name, blocks in abn_blocks_by_name.items():
            spec = spec_for_residue_name(residue_name, sgnn_specs)
            template_pdb = os.path.join(base_dir, 'pdb_bank', f'{residue_name}.pdb')
            bundle = build_sgnn_model_bundle(template_pdb, spec)
            abn_bundles[residue_name] = bundle
            abn_batch_forward[residue_name] = vmap(
                bundle.model.forward,
                in_axes=(0, None, None),
                out_axes=(0),
            )

        def dmff_calculator(pos, L, pairs, valid_pairs, valid_mask, atype_indices):
            box = jnp.array([[L,0,0],[0,L,0],[0,0,L]])
            E_nb = efunc_nb(pos, box, pairs, params_nb)
            E_bond = jnp.array(0.0)

            if standard_bundle is not None:
                pos_else = pos[non_abn_indices]
                E_bond = E_bond + standard_bundle.model.forward(
                    pos_else * 10,
                    box * 10,
                    standard_bundle.params,
                )

            for residue_name, blocks in abn_blocks_by_name.items():
                pos_abn = stack_positions_for_blocks(pos, blocks)
                bundle = abn_bundles[residue_name]
                E_bond = E_bond + jnp.sum(
                    abn_batch_forward[residue_name](
                        pos_abn * 10,
                        box * 10,
                        bundle.params,
                    )
                )

            # E_nb_ml = model_nb.apply(params, pos*10, box*10, valid_pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices)
            return E_nb + E_bond

        self.calc_dmff = jit(value_and_grad(dmff_calculator,argnums=(0, 1)))

        # compile tot_force function
        energy, (grad, virial) = self.calc_dmff(pos, L, self.pairs, self.valid_pairs, self.valid_mask, self.atype_indices)
        print(energy, grad, virial)
        return

    def grad(self, crd, cell): # receive SI input, return SI values
        pos = np.array(crd*1e9) # convert to nanometer
        box = np.array(cell*1e9) # convert to nanometer
        L = box[0][0]

        # nb list
        self.nbl.update(pos, box)
        pairs = self.nbl.pairs
        valid_pairs, valid_mask = filter_and_pad_pairs(pairs, self.atype_indices, max_pairs=self.max_pairs)

        energy, (grad, virial) = self.calc_dmff(pos, L, pairs, valid_pairs, valid_mask, self.atype_indices)
        virial = np.diag((-grad * pos).sum(axis=0) - virial*L/3).ravel()

        energy = np.array((energy*kilojoule_per_mole/AVOGADRO_CONSTANT_NA).value_in_unit(joule))
        grad = np.array((grad*kilojoule_per_mole/nanometer/AVOGADRO_CONSTANT_NA).value_in_unit(joule/meter))
        virial = np.array((virial*kilojoule_per_mole/AVOGADRO_CONSTANT_NA).value_in_unit(joule))
        return energy, grad, virial


if __name__ == '__main__':
    # the forces are composed by three parts: 
    # the long range part computed using openmm, parameters in xml
    # the short range part writen by hand, parameters in psr
    addr = sys.argv[1]
    port = int(sys.argv[2])
    socktype = sys.argv[3]

    driver_dmff = DMFFDriver(addr, port, socktype)
    while True:
        driver_dmff.parse()
