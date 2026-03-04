import torch
import torch.nn as nn
import numpy as np

@torch.jit.script
def pbc_shift(dr, box, box_inv):
    shift = torch.round(torch.matmul(dr, box_inv))
    return dr - torch.matmul(shift, box)

@torch.jit.script
def cutoff_cosine(r, rc : float):
    cutoff = 0.5 * (1.0 + torch.cos(torch.pi * r / rc))
    return torch.where(r <= rc, cutoff, torch.zeros_like(r))

@torch.jit.script
def get_topology_neighbors(pairs, topo_nblist, topo_mask):
    j_centers = pairs[:, 0]
    k_centers = pairs[:, 1]
    
    j_neighbors = topo_nblist[j_centers] # [n_pairs, max_neighbors]
    k_neighbors = topo_nblist[k_centers] 

    valid_j = j_neighbors != -1
    valid_k = k_neighbors != -1

    mask_j = (j_neighbors != j_centers.unsqueeze(-1)) & (j_neighbors != k_centers.unsqueeze(-1)) & valid_j
    mask_k = (k_neighbors != j_centers.unsqueeze(-1)) & (k_neighbors != k_centers.unsqueeze(-1)) & valid_k

    topo_mask_j = topo_mask[j_centers]
    topo_mask_k = topo_mask[k_centers]

    valid_mask_j = topo_mask_j & mask_j
    valid_mask_k = topo_mask_k & mask_k

    return j_neighbors, k_neighbors, valid_mask_j, valid_mask_k

class NeuralNetworkTorch(nn.Module):
    def __init__(self, in_features: int, dense_nodes: int = 64):
        super().__init__()
        self.dense_nodes = dense_nodes
        
        self.layers = nn.ModuleList()
        dim = in_features
        for _ in range(3):
            self.layers.append(nn.Linear(dim, dense_nodes))
            self.layers.append(nn.LayerNorm(dense_nodes))
            self.layers.append(nn.ReLU())
            dim = dense_nodes
        self.out_layer = nn.Linear(dim, 1)

    def forward(self, x, buffer_nblist_inter):
        for layer in self.layers:
            x = layer(x)
        out_AB = self.out_layer(x).squeeze(-1)
        energy = torch.sum(out_AB * buffer_nblist_inter)
        return energy

class FeatureExtractorTorch(nn.Module):
    def __init__(self, n_atoms: int, n_atype: int, rc: float, zindex: list, acsf_nmu: int = 20,
                 apsf_nmu: int = 10, acsf_eta: float = 100, apsf_eta: float = 25):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_atype = n_atype
        self.rc = rc
        self.acsf_nmu = acsf_nmu
        self.apsf_nmu = apsf_nmu
        self.acsf_eta = acsf_eta
        self.apsf_eta = apsf_eta
        
        self.register_buffer("acsf_mus", torch.linspace(0.0, 5.0, acsf_nmu))
        self.register_buffer("apsf_mus", torch.linspace(-1.0, 1.0, apsf_nmu))
        self.register_buffer("zindex_tensor", torch.tensor(zindex))
        # Map atomic numbers to 0..n_atype-1
        mapping = torch.zeros(int(max(zindex)) + 1, dtype=torch.long)
        for i, z in enumerate(zindex):
            mapping[int(z)] = i
        self.register_buffer("charge_mapping", mapping)

    def compute_atomcenter_features(self, pos, box, box_inv, topo_nblist, topo_mask, atype_indices):
        r_center = pos # [n_atoms, 3]
        safe_topo_nblist = torch.where(topo_nblist >= 0, topo_nblist, torch.zeros_like(topo_nblist))
        r_env = pos[safe_topo_nblist] # [n_atoms, max_neighbors, 3]
        
        dr = r_env - r_center.unsqueeze(1)
        dr = pbc_shift(dr, box, box_inv)
        dr_norm = torch.linalg.norm(dr + 1e-10, dim=2)
        
        f_cut = cutoff_cosine(dr_norm, self.rc) * topo_mask
        exp_term = torch.exp(-self.acsf_eta * torch.square(dr_norm.unsqueeze(-1) - self.acsf_mus))
        G_raw = exp_term * f_cut.unsqueeze(-1)
        
        type_one_hot = (atype_indices[safe_topo_nblist].unsqueeze(-1) == torch.arange(self.n_atype, device=pos.device))
        
        G = torch.einsum('ijk,ijl->ikl', G_raw, type_one_hot.float())
        return G

    def compute_atompair_features(self, cos_gamma_i, cos_gamma_j, j_list, k_list, j_mask, k_mask,
                                  buffer_nblist_inter_rc, atype_indices):
        angle_features_i = torch.exp(-self.apsf_eta * torch.square(cos_gamma_i.unsqueeze(-1) - self.apsf_mus))
        angle_features_j = torch.exp(-self.apsf_eta * torch.square(cos_gamma_j.unsqueeze(-1) - self.apsf_mus))

        safe_j = torch.where(j_list >= 0, j_list, torch.zeros_like(j_list))
        safe_k = torch.where(k_list >= 0, k_list, torch.zeros_like(k_list))

        type_one_hot_i = (atype_indices[safe_j].unsqueeze(-1) == torch.arange(self.n_atype, device=atype_indices.device))
        type_one_hot_j = (atype_indices[safe_k].unsqueeze(-1) == torch.arange(self.n_atype, device=atype_indices.device))

        masked_features_i = angle_features_i * j_mask.unsqueeze(-1).float()
        masked_features_j = angle_features_j * k_mask.unsqueeze(-1).float()

        G_i = torch.einsum('ijk,ijl->ikl', masked_features_i, type_one_hot_i.float())
        G_j = torch.einsum('ijk,ijl->ikl', masked_features_j, type_one_hot_j.float())

        G = (G_i + G_j) * 0.5 * buffer_nblist_inter_rc.unsqueeze(-1).unsqueeze(-1)
        return G

    def forward(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        box_inv = torch.linalg.inv(box)
        
        ri = pos[pairs[:, 0]]
        rj = pos[pairs[:, 1]]
        
        rij = rj - ri
        rij = pbc_shift(rij, box, box_inv)
        dr_norm = torch.linalg.norm(rij + 1e-10, dim=1)
        
        same_mol = mol_ID[pairs[:, 0]] == mol_ID[pairs[:, 1]]
        buffer_inter = (~same_mol).float()
        
        cutoff = 0.5 * (1.0 + torch.cos(torch.pi * dr_norm / self.rc))
        cutoff = torch.where(dr_norm <= self.rc, cutoff, torch.zeros_like(dr_norm))
        
        buffer_scales = valid_mask.float()
        # Not using mscales here assuming plain pairs
        buffer_nblist_inter = buffer_inter * buffer_scales
        buffer_nblist_inter_rc = buffer_nblist_inter * cutoff
        
        j_list, k_list, j_mask, k_mask = get_topology_neighbors(pairs, topo_nblist, topo_mask)
        
        valid_j_mask = j_mask.unsqueeze(-1)
        safe_j_list = torch.where(j_list >= 0, j_list, torch.zeros_like(j_list))
        rj_env = torch.where(valid_j_mask, pos[safe_j_list], torch.zeros_like(pos[0]))
        rj_X = pbc_shift(rj_env - ri.unsqueeze(1), box, box_inv)
        norm_rj_X = torch.linalg.norm(rj_X + 1e-10, dim=2, keepdim=True)
        rj_X_norm = rj_X / norm_rj_X
        rij_unit = rij / (dr_norm.unsqueeze(1) + 1e-10)
        cos_gamma_i = torch.einsum('aji,ai->aj', rj_X_norm, rij_unit) * j_mask.float()
        
        valid_k_mask = k_mask.unsqueeze(-1)
        safe_k_list = torch.where(k_list >= 0, k_list, torch.zeros_like(k_list))
        rk_env = torch.where(valid_k_mask, pos[safe_k_list], torch.zeros_like(pos[0]))
        rk_X = pbc_shift(rk_env - rj.unsqueeze(1), box, box_inv)
        norm_rk_X = torch.linalg.norm(rk_X + 1e-10, dim=2, keepdim=True)
        rk_X_norm = rk_X / norm_rk_X
        rji_unit = -rij_unit
        cos_gamma_j = torch.einsum('aji,ai->aj', rk_X_norm, rji_unit) * k_mask.float()
        
        atompair_features = self.compute_atompair_features(
            cos_gamma_i, cos_gamma_j, j_list, k_list, j_mask, k_mask,
            buffer_nblist_inter_rc, atype_indices
        )

        atom_features = self.compute_atomcenter_features(
            pos, box, box_inv, topo_nblist, topo_mask, atype_indices
        )
        
        atom_features_i = atom_features[pairs[:, 0]]
        atom_features_j = atom_features[pairs[:, 1]]
        atom_features = (atom_features_i + atom_features_j) * 0.5
        
        # Type one-hot features
        elem_indices = self.zindex_tensor[atype_indices]
        j_atype = elem_indices[pairs[:, 0]]
        k_atype = elem_indices[pairs[:, 1]]
        
        j_onehot_bits = torch.nn.functional.one_hot(self.charge_mapping[j_atype.long()], num_classes=self.n_atype).float()
        j_onehot = torch.cat([j_atype.unsqueeze(1).float(), j_onehot_bits], dim=1)
        
        k_onehot_bits = torch.nn.functional.one_hot(self.charge_mapping[k_atype.long()], num_classes=self.n_atype).float()
        k_onehot = torch.cat([k_atype.unsqueeze(1).float(), k_onehot_bits], dim=1)
        atype_onehot = torch.cat([j_onehot, k_onehot], dim=1)
        
        atom_features = atom_features.reshape(atom_features.shape[0], -1)
        atompair_features = atompair_features.reshape(atompair_features.shape[0], -1)
        
        apsf_features = torch.cat((atom_features, atompair_features, atype_onehot), dim=1)
        return apsf_features, dr_norm, buffer_nblist_inter_rc

class EAPNNForceTorch(nn.Module):
    def __init__(self, n_atoms: int, n_atype: int, rc: float, zindex: list, acsf_nmu: int = 20,
                 apsf_nmu: int = 10, acsf_eta: float = 100, apsf_eta: float = 25):
        super().__init__()
        self.zindex = zindex
        self.feature_extractor = FeatureExtractorTorch(
            n_atoms, n_atype, rc, zindex, acsf_nmu, apsf_nmu, acsf_eta, apsf_eta
        )
        
        # Fixed one-hot width based on original code:
        in_features = acsf_nmu * n_atype + apsf_nmu * n_atype + 22
        self.neural_network = NeuralNetworkTorch(in_features=in_features)

    def forward(self, pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        features, dr_norm, buffer_scales = self.feature_extractor(
            pos, box, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices
        )
        energy = self.neural_network(features, buffer_scales)
        return energy

class sGNNForceTorch(nn.Module):
    def __init__(self, G, n_layers=(3, 2), sizes=[(40, 20, 20), (20, 10)], nn_hops=1, sigma=162.13, mu=117.42):
        super().__init__()
        import numpy as np
        self.sigma = sigma
        self.mu = mu
        self.nn_hops = nn_hops
        self.max_valence = getattr(G, 'max_valence', 4)
        
        def to_tensor_float(x): return nn.Parameter(torch.tensor(np.array(x), dtype=torch.float32), requires_grad=False)
        def to_tensor_long(x): return nn.Parameter(torch.tensor(np.array(x), dtype=torch.long), requires_grad=False)
        
        self.bonds = to_tensor_long(G.bonds)
        self.b0 = to_tensor_float(G.b0)
        self.fscale_bond = G.fscale_bond
        
        self.angles = to_tensor_long(G.angles)
        self.cos_a0 = to_tensor_float(G.cos_a0)
        self.fscale_angle = G.fscale_angle
        
        self.diheds = to_tensor_long(G.diheds)
        
        self.feature_atypes = to_tensor_float(G.feature_atypes)
        self.indices_bonds = to_tensor_long(G.feature_indices['bonds'])
        self.indices_angles0 = to_tensor_long(G.feature_indices['angles0'])
        self.indices_angles1 = to_tensor_long(G.feature_indices['angles1'])
        self.indices_diheds = to_tensor_long(G.feature_indices['diheds'])
        
        if self.nn_hops == 1:
            self.nb_connect = to_tensor_float(G.nb_connect)
        self.weights = to_tensor_float(G.weights)
        
        self.w = nn.Parameter(torch.randn(1))
        
        self.fc0 = nn.ModuleList()
        dim_in = int(G.n_features)
        for i_layer in range(n_layers[0]):
            dim_out = int(sizes[0][i_layer])
            self.fc0.append(nn.Linear(dim_in, dim_out))
            dim_in = dim_out
            
        self.fc1 = nn.ModuleList()
        for i_layer in range(n_layers[1]):
            dim_out = int(sizes[1][i_layer])
            self.fc1.append(nn.Linear(dim_in, dim_out))
            dim_in = dim_out
            
        self.fc_final = nn.Linear(dim_in, 1)

    def calc_internal_coords_features(self, pos, box):
        box_inv = torch.linalg.inv(box)
        
        if len(self.bonds) > 0:
            pos0 = pos[self.bonds[:, 0]]
            pos1 = pos[self.bonds[:, 1]]
            dr = pos1 - pos0
            dr = pbc_shift(dr, box, box_inv)
            blength = torch.linalg.norm(dr, dim=1)
            fb = (blength - self.b0) * self.fscale_bond
        else:
            fb = torch.zeros(0, device=pos.device)
            
        if len(self.angles) > 0:
            rj = pos[self.angles[:, 0]]
            ri = pos[self.angles[:, 1]]
            rk = pos[self.angles[:, 2]]
            r_ij = pbc_shift(rj - ri, box, box_inv)
            r_ik = pbc_shift(rk - ri, box, box_inv)
            n_ij = torch.linalg.norm(r_ij, dim=1)
            n_ik = torch.linalg.norm(r_ik, dim=1)
            cos_a = torch.sum(r_ij * r_ik, dim=1) / (n_ij * n_ik + 1e-10)
            fa = (cos_a - self.cos_a0) * self.fscale_angle
        else:
            fa = torch.zeros(0, device=pos.device)
            
        if len(self.diheds) > 0:
            ri = pos[self.diheds[:, 0]]
            rj = pos[self.diheds[:, 1]]
            rk = pos[self.diheds[:, 2]]
            rl = pos[self.diheds[:, 3]]
            r_jk = pbc_shift(rk - rj, box, box_inv)
            r_ji = pbc_shift(ri - rj, box, box_inv)
            r_kl = pbc_shift(rl - rk, box, box_inv)
            r_kj = -r_jk
            n1 = torch.cross(r_jk, r_ji, dim=1)
            n2 = torch.cross(r_kl, r_kj, dim=1)
            norm_n1 = torch.linalg.norm(n1, dim=1)
            norm_n2 = torch.linalg.norm(n2, dim=1)
            fd = torch.sum(n1 * n2, dim=1) / (norm_n1 * norm_n2 + 1e-10)
        else:
            fd = torch.zeros(0, device=pos.device)
            
        return fb, fa, fd

    def forward(self, pos, box):
        fb, fa, fd = self.calc_internal_coords_features(pos, box)
        
        fb_pad = torch.cat([fb, torch.zeros(1, device=fb.device)])
        fa_pad = torch.cat([fa, torch.zeros(1, device=fa.device)])
        fd_pad = torch.cat([fd, torch.zeros(1, device=fd.device)])
        
        idx_b = torch.where(self.indices_bonds == -1, len(fb), self.indices_bonds)
        idx_a0 = torch.where(self.indices_angles0 == -1, len(fa), self.indices_angles0)
        idx_a1 = torch.where(self.indices_angles1 == -1, len(fa), self.indices_angles1)
        idx_d = torch.where(self.indices_diheds == -1, len(fd), self.indices_diheds)
        
        f_bonds = fb_pad[idx_b] * (self.indices_bonds >= 0).float()
        f_angles0 = fa_pad[idx_a0] * (self.indices_angles0 >= 0).float()
        f_angles1 = fa_pad[idx_a1] * (self.indices_angles1 >= 0).float()
        f_diheds = fd_pad[idx_d] * (self.indices_diheds >= 0).float()
        
        features = torch.cat([self.feature_atypes, f_bonds, f_angles0, f_angles1, f_diheds], dim=-1)
        
        for layer in self.fc0:
            features = torch.tanh(layer(features))
            
        if self.nn_hops == 1:
            mv = self.max_valence
            nb_connect0 = self.nb_connect[..., :mv-1]
            nb_connect1 = self.nb_connect[..., mv-1:2*(mv-1)]
            nb0 = torch.sum(nb_connect0, dim=-1, keepdim=True)
            nb1 = torch.sum(nb_connect1, dim=-1, keepdim=True)
            
            f_center = features[:, 0, :]
            f_nb0 = features[:, 1:mv, :]
            f_nb1 = features[:, mv:2*mv-1, :]
            
            sum_nb0 = torch.bmm(nb_connect0.unsqueeze(1), f_nb0).squeeze(1)
            sum_nb1 = torch.bmm(nb_connect1.unsqueeze(1), f_nb1).squeeze(1)
            
            w = self.w
            h0 = (nb0 > 0.5).float()
            h1 = (nb1 > 0.5).float()
            
            term1 = f_center * (1.0 - h0 * w - h1 * w)
            term2 = (w * sum_nb0) / torch.where(nb0 < 1e-5, torch.ones_like(nb0)*1e-5, nb0)
            term3 = (w * sum_nb1) / torch.where(nb1 < 1e-5, torch.ones_like(nb1)*1e-5, nb1)
            features = term1 + term2 + term3
        else:
            features = features[:, 0, :]
            
        for layer in self.fc1:
            features = torch.tanh(layer(features))
            
        energies = self.fc_final(features).squeeze(-1)
        total_energy = torch.sum(self.weights * energies) * self.sigma + self.mu
        return total_energy
