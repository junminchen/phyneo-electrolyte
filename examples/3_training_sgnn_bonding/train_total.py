#!/usr/bin/env python
import sys
import jax
import time
import jax.numpy as jnp
from jax import grad, value_and_grad, jit
import numpy as np
import dmff
from dmff.utils import jit_condition
from dmff.sgnn.gnn import MolGNNForce
from dmff.sgnn.graph import TopGraph, from_pdb

import optax
import pickle
# use pytorch data loader
from torch.utils.data import DataLoader

from jax.lib import xla_bridge 
print(jax.devices()[0]) 
print(xla_bridge.get_backend().platform)

class MolDataSet():
    def __init__(self, data):
        self.data = data
        self.n_data = len(self.data['positions'])
        return

    def __getitem__(self, i):
        return [self.data['positions'][i], self.data['energies'][i]]

    def __len__(self):
        return self.n_data

if __name__ == "__main__":
    box = jnp.eye(3) * 50

    with open('../dataset/data_300k_remove_nb.pickle', 'rb') as ifile:
        tot_data = pickle.load(ifile)
    with open('../dataset/data_1000k_remove_nb.pickle', 'rb') as ifile:
        tot_data_ = pickle.load(ifile)
        
    # monomer_train = ['conf_01_DEC', 'conf_02_DFEA', 'conf_03_DMC', 'conf_04_DME', 'conf_05_EMC', 'conf_06_EP', 'conf_07_FEMC', 
    #                     'conf_08_PP', 'conf_09_EC', 'conf_10_PC', 'conf_11_PS', 'conf_12_SL', 'conf_13_FEC', 'conf_14_GBL', 
    #                     'conf_22_CN1', 'conf_23_CN2',
    #                     'conf_16_BOB', 'conf_17_DFOB', 'conf_19_FSI', 'conf_21_TFSI', 
    #                     # 'conf_18_DFP', 'conf_20_PF6', 'conf_15_BF4', 
    #                     ]

    monomer_train = ['conf_01_DEC', 'conf_02_DFEA', 'conf_03_DMC', 'conf_04_DME', 'conf_05_EMC', 'conf_06_EP', 'conf_07_FEMC', 
                        'conf_08_PP', 'conf_09_EC', 'conf_10_PC', 'conf_11_PS', 'conf_12_SL', 'conf_13_FEC', 'conf_14_GBL', 
                        'conf_22_CN1', 'conf_23_CN2',
                        'conf_16_BOB', 'conf_17_DFOB', 'conf_19_FSI', 'conf_21_TFSI',]
    
    natoms = []                        
    # monomer_train = ['conf_04_DME']
    trunk_train = []
    trunk_test = {}
    cal_energy = {}
    MSELoss_grad = {}
    # for key in list(tot_data.keys()):
    for key in monomer_train:
        print(key)
        data = tot_data[key]
        natoms.append(data['positions'].shape[1])
        print(data['positions'].shape[1])
        trunk_test[key] = {}
        data_train = {}
        for comp in ['positions', 'energies']:
            data_train[comp] = data[comp][:int(0.9*len(data[comp]))]
            trunk_test[key][comp] = data[comp][int(0.9*len(data[comp])):]

        # training and testing data
        dataset = MolDataSet(data_train)
        train_loader = DataLoader(dataset, shuffle=True, batch_size=64)
        for ibatch, (pos, e) in enumerate(train_loader):
            pos = jnp.array(pos.numpy())
            ene_ref = jnp.array(e.numpy())
            trunk_train.append([key, pos, ene_ref])

        data = tot_data_[key]
        for comp in ['positions', 'energies']:
            data_train[comp] = data[comp][:int(0.9*len(data[comp]))]        
        # training and testing data
        dataset = MolDataSet(data_train)
        train_loader = DataLoader(dataset, shuffle=True, batch_size=64)
        for ibatch, (pos, e) in enumerate(train_loader):
            pos = jnp.array(pos.numpy())
            ene_ref = jnp.array(e.numpy())
            trunk_train.append([key, pos, ene_ref])

        pdb = f'pdb_bank/{key.split("_")[-1]}.pdb'
        # Graph and model
        G = from_pdb(pdb)
        model = MolGNNForce(G, nn=1)
        cal_energy[key] = jax.vmap(model.forward, in_axes=(0, None, None), out_axes=(0))
        def MSELoss(params, positions, box, ene_ref):
            ene = cal_energy[key](positions, box, params)
            err = ene - ene_ref
            # we do not care about constant shifts
            err -= jnp.average(err)
            return jnp.average(err**2)
        MSELoss_grad[key] = jit(value_and_grad(MSELoss, argnums=(0)))

    restart = 'params_total/params_sgnn.pickle'
    # restart = None
    if restart is not None:
        with open(restart, 'rb') as f:
            params = pickle.load(f)    
    else:
        params = model.params
    
    # optmizer
    lr = 0.001
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # train
    best_loss = jnp.array(1e30)
    n_epochs = 3000
    fout=open('nn.err','w')
    fout.write("Sub-Graph Nerual Network Package used for intramolecular energy\n")
    fout.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
    fout.flush()
    for i_epoch in range(n_epochs):
        np.random.shuffle(trunk_train)
        # train an epoch
        lossprop = 0
        for key, pos, ene_ref in trunk_train:
            loss, gradients = MSELoss_grad[key](params, pos, box, ene_ref)
            lossprop += loss
            updates, opt_state = optimizer.update(gradients, opt_state)
            params = optax.apply_updates(params, updates)
        lossprop = jnp.sqrt(lossprop)
        print(lossprop)        
        
        if lossprop < best_loss:
            # save model after each epoch
            with open('params_sgnn.pickle', 'wb') as f:
                pickle.dump(params, f)            
            # model.save_params('params_sgnn.pickle') 
            best_loss = lossprop      

            # evaluate test
            ene_refs = []
            ene_preds = []
            test_loss = []
            for key in trunk_test:
                ene_pred = cal_energy[key](trunk_test[key]['positions'], box, params)
                ene_ref = trunk_test[key]['energies']
                loss_mol = jnp.average((ene_ref - jnp.average(ene_ref) - (ene_pred - jnp.average(ene_pred)))**2)
                loss_mol = jnp.sqrt(loss_mol)
                print(key, loss_mol)
                ene_preds.append(ene_pred - jnp.average(ene_pred))
                ene_refs.append(ene_ref - jnp.average(ene_ref))
                
            ene_ref_tot = jnp.concatenate(ene_refs) - jnp.average(jnp.concatenate(ene_refs))
            ene_pred_tot = jnp.concatenate(ene_preds) - jnp.average(jnp.concatenate(ene_preds))
            err = ene_pred_tot - ene_ref_tot
            test_loss = jnp.sqrt(jnp.average(err**2))

            fout.write("{:5} {:4} {:15} {:5e}  {} ".format("Epoch=",i_epoch,"learning rate",lr,"train error:"))
            fout.write('{:10.5f} '.format(lossprop))
            fout.write('{} '.format("test error:"))
            fout.write('{:10.5f} \n'.format(test_loss))
            fout.flush()
            # print test data
            with open('test_data.xvg', 'w') as f:
                print('# RMSE = %10.5f'%test_loss, file=f)
                for e1, e2 in zip(ene_pred_tot, ene_ref_tot):
                    print(e2-np.mean(ene_ref_tot), e2-np.mean(ene_ref_tot), e1-np.mean(ene_pred_tot), file=f)
            
    fout.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
    fout.write("terminated normal\n")
    fout.close()
