import pickle
import re
import sys
from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np
from graph_torch import MAX_VALENCE, TopGraph, from_pdb
# import jax.numpy as jnp


'''
def prm_transform_f2i(params, n_layers):
    p = {}
    for k in params:
        if isinstance(params[k], np.ndarray):
            p[k] = torch.tensor(params[k])
        elif isinstance(params[k], torch.Tensor):
            p[k] = params[k]
        else:
            p[k] = params[k]

    for i_nn in [0, 1]:
        nn_name = 'fc%d' % i_nn
        p['%s.weight' % nn_name] = []
        p['%s.bias' % nn_name] = []
        for i_layer in range(n_layers[i_nn]):
            k_w = '%s.%d.weight' % (nn_name, i_layer)
            k_b = '%s.%d.bias' % (nn_name, i_layer)
            p['%s.weight' % nn_name].append(p.pop(k_w, None))
            p['%s.bias' % nn_name].append(p.pop(k_b, None))
'''


def prm_transform_f2i(params, n_layers):
    p = {}
    for k in params:
        if isinstance(params[k], np.ndarray):
            p[k] = torch.tensor(params[k], dtype=torch.float32)
        elif isinstance(params[k], torch.Tensor):
            # p[k] = params[k]
            p[k] = torch.tensor(params[k], dtype=torch.float32)
        else:
            p[k] = params[k]
    return p


def prm_transform_i2f(params, n_layers):
    p = {}
    p['w'] = params['w']
    p['fc_final.weight'] = params['fc_final.weight']
    p['fc_final.bias'] = params['fc_final.bias']
    for i_nn in range(2):
        nn_name = 'fc%d' % i_nn
        for i_layer in range(n_layers[i_nn]):
            p[nn_name + '.%d.weight' %
                   i_layer] = params[nn_name + '.weight'][i_layer]
            p[nn_name +
                   '.%d.bias' % i_layer] = params[nn_name +
                                                  '.bias'][i_layer]
    return p


class MolGNNForce(torch.nn.Module):

    def __init__(self, G, n_layers=(3, 2), sizes=[(40, 20, 20), (20, 10)], 
                  nn=1, sigma=162.13039087945623, 
                 mu=117.41975505778706, seed=12345, device='cpu'):
        super(MolGNNForce, self).__init__()
        
        
        self.device = torch.device(device)
        self.nn = nn
        self.G = G
        self.G.get_all_subgraphs(nn, typify=True)
        self.G.prepare_subgraph_feature_calc()
        
        torch.manual_seed(seed)
        
        self.n_layers = n_layers
        self.sizes = sizes
        self.sigma = sigma
        self.mu = mu
        
        self._w = torch.nn.Parameter(torch.rand(1,  device=self.device))
        
        dim_in = G.n_features
        dim_in = int(dim_in)
        
        self.fc0_layers = torch.nn.ModuleList()
        self.fc1_layers = torch.nn.ModuleList()
        
        for i_layer in range(n_layers[0]):
            dim_out = int(sizes[0][i_layer])
            layer = torch.nn.Linear(dim_in, dim_out)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='tanh')
            torch.nn.init.zeros_(layer.bias)
            self.fc0_layers.append(layer)
            dim_in = dim_out
        
        for i_layer in range(n_layers[1]):
            dim_out = int(sizes[1][i_layer])
            layer = torch.nn.Linear(dim_in, dim_out)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='tanh')
            torch.nn.init.zeros_(layer.bias)
            self.fc1_layers.append(layer)
            dim_in = dim_out
        
        self.fc_final = torch.nn.Linear(dim_in, 1)
        torch.nn.init.kaiming_uniform_(self.fc_final.weight)
        torch.nn.init.uniform_(self.fc_final.bias)
        
        self.to(self.device)
        
        return

    def fc0(self, f_in):
        f = f_in
        for layer in self.fc0_layers:
            f = torch.tanh(layer(f))
        return f

    def fc1(self, f_in):
        f = f_in
        for layer in self.fc1_layers:
            f = torch.tanh(layer(f))
        return f

    def message_pass(self, f_in, nb_connect, w, nn):
        if nn == 0:
            return f_in[:, 0, :]
        elif nn == 1:
            nb_connect0 = nb_connect[:, :MAX_VALENCE - 1]  # Shape: (n_subgraphs, 3)
            nb_connect1 = nb_connect[:, MAX_VALENCE - 1:2 * (MAX_VALENCE - 1)]  # Shape: (n_subgraphs, 3)
            nb0 = torch.sum(nb_connect0, dim=1, keepdim=True).to(f_in.dtype)
            nb1 = torch.sum(nb_connect1, dim=1, keepdim=True).to(f_in.dtype)
            
            f_center = f_in[:, 0, :]  
            f_nb0 = f_in[:, 1:MAX_VALENCE, :]  
            f_nb1 = f_in[:, MAX_VALENCE:2*MAX_VALENCE-1, :]  
            
            weighted_nb0 = torch.sum(nb_connect0.unsqueeze(-1) * f_nb0, dim=1).to(f_in.dtype)
            weighted_nb1 = torch.sum(nb_connect1.unsqueeze(-1) * f_nb1, dim=1).to(f_in.dtype)

            nb0_safe = torch.where(nb0 < 1e-5, torch.tensor(1e-5, device=self.device), nb0)
            nb1_safe = torch.where(nb1 < 1e-5, torch.tensor(1e-5, device=self.device), nb1)
            
            heaviside_nb0 = (nb0 > 0)
            heaviside_nb1 = (nb1 > 0)
            
            f = f_center * (1 - heaviside_nb0 * w - heaviside_nb1 * w) + \
                w * weighted_nb0 / nb0_safe + \
                w * weighted_nb1 / nb1_safe
            
            return f

    def forward(self, positions, box=None):
        if isinstance(positions, torch.Tensor):
            positions = positions.to(device=self.device)
        else:
            positions = torch.tensor(positions, device=self.device)
        
        if box is None:
            box = self.G.box
            
        if box is not None:
            if isinstance(box, torch.Tensor):
                box = box.to(device=self.device)
            else:
                box = torch.tensor(box, device=self.device)
        
        features = self.G.calc_subgraph_features(positions, box)
        features = self.fc0(features)
        features = self.message_pass(features, self.G.nb_connect, self.w, self.G.nn)
        features = self.fc1(features)
        
        energies = self.fc_final(features)
        energies = energies.squeeze(-1)
        total_energy = torch.sum(self.G.weights * energies) * self.sigma + self.mu
        return total_energy

    def batch_forward(self, positions_batch, box_batch):
        batch_size = positions_batch.shape[0]
        energies = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            energies[i] = self.forward(positions_batch[i], box_batch[i])
        return energies

    def get_energy(self, positions, box=None, params=None):
        if box is None:
            box = self.G.box
            
        return self.forward(positions, box)

    @property
    def params(self):
        return self.parameters_dict()

    @property
    def w(self):
        return getattr(self, '_w', torch.nn.Parameter(torch.rand(1, device=self.device)))
    
    @w.setter
    def w(self, value):
        # if isinstance(value, torch.Tensor):
        #     self._w = torch.nn.Parameter(value.to(device=self.device))
        # else:
        #     self._w = torch.nn.Parameter(torch.tensor(value, device=self.device))
        if isinstance(value, torch.Tensor):
            self._w.copy_(value.to(device=self.device))
        else:
            self._w.copy_(torch.tensor(value, device=self.device))

    def load_params(self, ifn):
        with open(ifn, 'rb') as ifile:
            params = pickle.load(ifile)
        
        for k in params.keys():
            if isinstance(params[k], torch.Tensor):
                params[k] = params[k].to(device=self.device)
            elif isinstance(params[k], torch.Tensor):
                params[k] = torch.tensor(params[k], device=self.device)
        
        params_internal = prm_transform_f2i(params, self.n_layers)
        
        for k in params_internal.keys():
            if isinstance(params_internal[k], torch.Tensor):
                params_internal[k] = params_internal[k].to(device=self.device)
            elif isinstance(params_internal[k], list):
                for i, item in enumerate(params_internal[k]):
                    if isinstance(item, torch.Tensor):
                        params_internal[k][i] = item.to(device=self.device)
        
        with torch.no_grad():
            self.w = params_internal['w'].to(device=self.device)
            
            for i, layer in enumerate(self.fc0_layers):
                if i < len(params_internal['fc0.weight']):
                    saved_weight = params_internal['fc0.weight'][i]
                    saved_bias = params_internal['fc0.bias'][i]

                    if saved_weight.shape == layer.weight.shape:
                        layer.weight.copy_(saved_weight)
                    if saved_bias.shape == layer.bias.shape:
                        layer.bias.copy_(saved_bias)
            
            for i, layer in enumerate(self.fc1_layers):
                if i < len(params_internal['fc1.weight']):
                    saved_weight = params_internal['fc1.weight'][i]
                    saved_bias = params_internal['fc1.bias'][i]
                    if saved_weight.shape == layer.weight.shape:
                        layer.weight.copy_(saved_weight)
                    if saved_bias.shape == layer.bias.shape:
                        layer.bias.copy_(saved_bias)
            
            saved_final_weight = params_internal['fc_final.weight'].to(device=self.device)
            saved_final_bias = params_internal['fc_final.bias'].to(device=self.device)
            
            if saved_final_weight.shape == self.fc_final.weight.shape:
                self.fc_final.weight.copy_(saved_final_weight)
            if saved_final_bias.shape == self.fc_final.bias.shape:
                self.fc_final.bias.copy_(saved_final_bias)
        
        return

    def save_params(self, ofn):
        """ Save the network parameters to a pickle file

        Parameters
        ----------
        ofn: string
            the output file name

        """
        params_internal = OrderedDict()
        
        params_internal['w'] = self.w.detach().cpu()
        
        params_internal['fc0.weight'] = []
        params_internal['fc0.bias'] = []
        for layer in self.fc0_layers:
            params_internal['fc0.weight'].append(layer.weight.detach().cpu().T)
            params_internal['fc0.bias'].append(layer.bias.detach().cpu())
        
        params_internal['fc1.weight'] = []
        params_internal['fc1.bias'] = []
        for layer in self.fc1_layers:
            params_internal['fc1.weight'].append(layer.weight.detach().cpu().T)
            params_internal['fc1.bias'].append(layer.bias.detach().cpu())
        
        params_internal['fc_final.weight'] = self.fc_final.weight.detach().cpu()
        params_internal['fc_final.bias'] = self.fc_final.bias.detach().cpu()
        
        params = prm_transform_i2f(params_internal, self.n_layers)
        

        for k in params:
            if isinstance(params[k], torch.Tensor):
                params[k] = params[k].numpy()
        
        with open(ofn, 'wb') as ofile:
            pickle.dump(params, ofile)
        return

    def parameters_dict(self):
        params = OrderedDict()
        params['w'] = self.w
        for i, layer in enumerate(self.fc0_layers):
            params[f'fc0.{i}.weight'] = layer.weight.T
            params[f'fc0.{i}.bias'] = layer.bias
        for i, layer in enumerate(self.fc1_layers):
            params[f'fc1.{i}.weight'] = layer.weight.T
            params[f'fc1.{i}.bias'] = layer.bias
        params['fc_final.weight'] = self.fc_final.weight
        params['fc_final.bias'] = self.fc_final.bias
        return params

    def set_parameters_dict(self, params):
        with torch.no_grad():
            if 'w' in params:
                self.w = params['w']
            
            for i, layer in enumerate(self.fc0_layers):
                weight_key = f'fc0.{i}.weight'
                bias_key = f'fc0.{i}.bias'
                if weight_key in params:
                    layer.weight.copy_(params[weight_key].T)
                if bias_key in params:
                    layer.bias.copy_(params[bias_key])
            
            for i, layer in enumerate(self.fc1_layers):
                weight_key = f'fc1.{i}.weight'
                bias_key = f'fc1.{i}.bias'
                if weight_key in params:
                    layer.weight.copy_(params[weight_key].T)
                if bias_key in params:
                    layer.bias.copy_(params[bias_key])
            
            if 'fc_final.weight' in params:
                self.fc_final.weight.copy_(params['fc_final.weight'])
            if 'fc_final.bias' in params:
                self.fc_final.bias.copy_(params['fc_final.bias'])

    # def train(self):
    #     self.train()
    #     return self

    def train(self, mode=False):
        super().train(mode)
        return self
        
    def eval_mode(self):
        self.eval()
        return self

    def to_device(self, device):
        self.device = torch.device(device)
        self.to(self.device)
        if hasattr(self.G, 'device'):
            self.G.device = self.device
        return self

    def compute_gradients(self, positions, box=None):
        if box is None:
            box = self.G.box
            
        positions.requires_grad_(True)
        energy = self.forward(positions, box)
        gradients = torch.autograd.grad(energy, positions, create_graph=True)[0]
        return energy, -gradients  # Forces are negative gradients

    def energy_and_forces(self, positions, box):
        return self.compute_gradients(positions, box)


