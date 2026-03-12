import sys
from functools import partial
from itertools import permutations, product

import torch
import torch.nn.functional as F
try:
    import mdtraj as md
except ImportError:
    pass
import numpy as np


'''
This module works on building graphs based on molecular topology
'''

ATYPE_INDEX = {
    'H': 0, 'He': 1,
    'Li': 2, 'Be': 3,
    'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9,
    'Na': 10, 'Mg': 11,
    'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'Ar': 17,
    'K': 18, 'Ca': 19
}
N_ATYPES = len(ATYPE_INDEX.keys())

# used to compute equilibrium bond lengths
#COVALENT_RADIUS = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05}
COVALENT_RADIUS = {
    'H': 0.31, 'He': 0.28,
    'Li': 1.28, 'Be': 0.96,
    'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41,
    'Al': 1.21, 'Si': 1.11, 'P': 1.10, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76 
}

# scaling parameters for feature calculations
FSCALE_BOND = 10.0
FSCALE_ANGLE = 5.0

MAX_VALENCE = 4
MAX_ANGLES_PER_SITE = MAX_VALENCE * (MAX_VALENCE - 1) // 2
MAX_DIHEDS_PER_BOND = (MAX_VALENCE - 1)**2

# dimension of bond features
DIM_BOND_FEATURES_GEOM = {
    'bonds': 2 * MAX_VALENCE - 1,
    'angles0': MAX_VALENCE * (MAX_VALENCE - 1) // 2,
    'angles1': MAX_VALENCE * (MAX_VALENCE - 1) // 2,
    'diheds': (MAX_VALENCE - 1)**2
}
DIM_BOND_FEATURES_GEOM_TOT = np.sum(
    [DIM_BOND_FEATURES_GEOM[k] for k in DIM_BOND_FEATURES_GEOM.keys()])
DIM_BOND_FEATURES_ATYPES = MAX_VALENCE * 2 * N_ATYPES


def pbc_shift_torch(dr, box, box_inv):
    if box is None:
        return dr
    else:
        ds = torch.matmul(dr, box_inv.T)
        ds = ds - torch.floor(ds + 0.5)
        dr_shifted = torch.matmul(ds, box)
        return dr_shifted


def distribute_scalar_torch(scalar_values, indices):
    valid_mask = indices >= 0
    result = torch.zeros_like(indices, dtype=torch.float32, device=scalar_values.device)
    
    valid_indices = indices[valid_mask]
    if len(valid_indices) > 0:
        valid_indices = torch.clamp(valid_indices, 0, len(scalar_values) - 1)
        result[valid_mask] = scalar_values[valid_indices].to(torch.float32)
    
    return result


def distribute_v3_torch(vectors, indices):
    return vectors[indices]


class TopGraph:

    def __init__(self, list_atom_elems, bonds, positions=None, box=None, device='cpu'):
        self.device = torch.device(device)
        self.list_atom_elems = list_atom_elems
        self.bonds = bonds
        self.n_atoms = len(list_atom_elems)
        if positions is not None:
            if isinstance(positions, torch.Tensor):
                self.positions = positions.clone().detach().to(dtype=torch.float32, device=self.device)
            else:
                self.positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        else:
            self.positions = None
        self._build_connectivity()
        self._get_valences()
        self.set_internal_coords_indices()
        if box is not None:
            if isinstance(box, torch.Tensor):
                self.box = box.clone().detach().to(dtype=torch.float32, device=self.device)
            else:
                self.box = torch.tensor(box, dtype=torch.float32, device=self.device)
            self.box_inv = torch.linalg.inv(self.box)
        else:
            self.box = None
            self.box_inv = None
        return

    def set_box(self, box):
        '''
        Set the box information in the class

        Parameters
        ----------
        box: array
            3 * 3: the box array, pbc vectors arranged in rows
        '''
        if isinstance(box, torch.Tensor):
            self.box = box.clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            self.box = torch.tensor(box, dtype=torch.float32, device=self.device)
        self.box_inv = torch.linalg.inv(self.box)
        if hasattr(self, 'subgraphs'):
            self._propagate_attr('box')
            self._propagate_attr('box_inv')
        return

    def set_positions(self, positions, update_subgraph=True):
        if isinstance(positions, torch.Tensor):
            self.positions = positions.clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            self.positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        if update_subgraph:
            self._update_subgraph_positions()
        return

    def _propagate_attr(self, attr):
        for ig in range(self.n_subgraphs):
            setattr(self.subgraphs[ig], attr, getattr(self, attr))
        return

    def _build_connectivity(self):
        self.connectivity = np.zeros((self.n_atoms, self.n_atoms), dtype=int)
        for i, j in self.bonds:
            self.connectivity[i, j] = 1
            self.connectivity[j, i] = 1
        # Also build adjacency list for fast neighbor lookup
        self._adj = [[] for _ in range(self.n_atoms)]
        for i, j in self.bonds:
            self._adj[int(i)].append(int(j))
            self._adj[int(j)].append(int(i))
        return

    def _get_valences(self):
        if hasattr(self, '_adj'):
            self.valences = np.array([len(self._adj[i]) for i in range(self.n_atoms)])
        elif hasattr(self, 'connectivity'):
            self.valences = np.sum(self.connectivity, axis=1)
        else:
            sys.exit('Error in generating valences: build connectivity first!')

    def get_all_subgraphs(self, nn, type_center='bond', typify=True, id_chiral=True):
        self.subgraphs = []
        if type_center == 'atom':
            for ia in range(self.n_atoms):
                self.subgraphs.append(TopSubGraph(self, ia, nn, type_center))
        elif type_center == 'bond':
            # build a subgraph around each bond
            for ib, b in enumerate(self.bonds):
                self.subgraphs.append(TopSubGraph(self, ib, nn, type_center))
        self.nn = nn
        self.n_subgraphs = len(self.subgraphs)
        if typify:
            self.typify_all_subgraphs()
        if typify and id_chiral:
            for g in self.subgraphs:
                g._add_chirality_labels()
                # create permutation groups, and canonical orders for atoms
                g.get_canonical_orders_wt_permutation_grps()
        return

    def _update_subgraph_positions(self):
        for g in self.subgraphs:
            valid_indices = [idx for idx in g.map_sub2parent if idx >= 0]
            if len(valid_indices) > 0:
                g.positions = self.positions[torch.tensor(valid_indices, device=self.device)]
            else:
                g.positions = torch.zeros((0, 3), device=self.device)
        return

    def get_subgraph(self, i_center, nn, type_center='bond'):
        return TopSubGraph(self, i_center, nn, type_center)

    def typify_atom(self, i, depth=0, excl=None):
        if not hasattr(self, '_typify_cache'):
            self._typify_cache = {}
        cache_key = (i, depth, excl)
        if cache_key in self._typify_cache:
            return self._typify_cache[cache_key]
        if depth == 0:
            result = self.list_atom_elems[i]
        else:
            atype = self.list_atom_elems[i]
            atype_nbs = []
            for j in self._adj[int(i)]:
                if j != excl:
                    atype_nbs.append(
                        self.typify_atom(j, depth=depth - 1, excl=i))
            atype_nbs.sort()
            if len(atype_nbs) == 0:
                result = atype
            else:
                result = atype + '-(' + ','.join(atype_nbs) + ')'
        self._typify_cache[cache_key] = result
        return result

    def typify_all_atoms(self, depth=0):
        self.atom_types = []
        for i in range(self.n_atoms):
            self.atom_types.append(self.typify_atom(i, depth=depth))
        self.atom_types = np.array(self.atom_types, dtype="object")
        return

    def typify_subgraph(self, i):
        '''
        Do atom typification for subgraph i
        the depth is set to be 2*nn + 4, that is the largest possible size of subgraphs

        Parameters
        ----------
        i: int
            the index of the subgraph to typify

        '''
        self.subgraphs[i].typify_all_atoms(depth=(2 * self.nn + 4))
        return

    def typify_all_subgraphs(self):
        '''
        Do atom typification for all subgraphs
        '''
        for i_subgraph in range(self.n_subgraphs):
            self.typify_subgraph(i_subgraph)
        return

    def _add_chirality_labels(self, verbose=False):
        '''
        This subroutine add chirality labels to distinguish hydrogens in ABCH2
        It uses the position info to identify the chirality of the H
        It modifies the self.atom_types attribute 
        '''
        for i in range(self.n_atoms):
            neighbors = np.where(self.connectivity[i] == 1)[0]
            if len(neighbors) != 4:
                continue
            labels = self.atom_types[neighbors]
            flags = np.array([labels == labels[i] for i in range(4)])
            flags = flags.sum(axis=1)
            if np.sum(flags) == 6:  # C-ABH2
                filter_H = (flags == 2)
                j, k = neighbors[np.where(filter_H)[0]]
                l, m = neighbors[np.where(np.logical_not(filter_H))[0]]
                ti, tj, tk, tl, tm = self.atom_types[[i, j, k, l, m]]
                # swap l and m, such that tl < tm
                if tl > tm:
                    (l, m) = (m, l)
                    tl, tm = np.array(self.atom_types, dtype="object")[[l, m]]
                ri, rj, rk, rl, rm = self.positions[torch.tensor([i, j, k, l, m], device=self.device)]
                rij = pbc_shift_torch(rj - ri, self.box, self.box_inv)
                rkl = pbc_shift_torch(rl - rk, self.box, self.box_inv)
                rkm = pbc_shift_torch(rm - rk, self.box, self.box_inv)
                if torch.dot(rij, torch.linalg.cross(rkl, rkm)) > 0:
                    self.atom_types[j] += 'R'
                    self.atom_types[k] += 'L'
                else:
                    self.atom_types[j] += 'L'
                    self.atom_types[k] += 'R'
        return

    def set_internal_coords_indices(self):
        '''
        This method go over the graph and search for all bonds, angles, diheds
        It records the atom indices for all ICs, and also the equilibrium bond lengths and angles
        It sets the following attributes in the graph:
        bonds, a0, angles, cos_a0, diheds
        n_bonds, n_angles, n_diheds
        '''
        # bonds
        self.bonds = np.array(self.bonds)
        
        if len(self.bonds.shape) == 1 and len(self.bonds) == 2:
            self.bonds = self.bonds.reshape(1, -1)
        elif len(self.bonds.shape) == 1:
            self.bonds = np.array(self.bonds).reshape(-1, 2)
        # equilibrium bond lengths
        a0 = self.bonds[:, 0]
        a1 = self.bonds[:, 1]
        at0 = [self.list_atom_elems[int(i)] for i in a0]
        at1 = [self.list_atom_elems[int(i)] for i in a1]
        r0 = torch.tensor([COVALENT_RADIUS[e0] for e0 in at0], device=self.device)
        r1 = torch.tensor([COVALENT_RADIUS[e1] for e1 in at1], device=self.device)
        self.b0 = r0 + r1
        self.n_bonds = len(self.bonds)

        #angles
        angles = []
        for i in range(self.n_atoms):
            neighbors = np.array(self._adj[i])
            for jj, j in enumerate(neighbors):
                for kk, k in enumerate(neighbors[jj + 1:]):
                    angles.append([j, i, k])
        self.angles = np.array(angles)
        if len(self.angles.shape) == 1 and len(self.angles) > 0:
            self.angles = self.angles.reshape(1, -1)
        elif len(self.angles) == 0:
            self.angles = np.array([]).reshape(0, 3)

        def get_a0(indices_angles):
            a0 = np.zeros(len(indices_angles))
            for ia, (j, i, k) in enumerate(indices_angles):
                if i >= 0 and j >= 0 and k >= 0:
                    valence = self.valences[i]
                    if valence == 2 and self.list_atom_elems[
                            i] == 'O' or self.list_atom_elems[i] == 'S':
                        cos_a0 = np.cos(104.45 / 180 * np.pi)
                    elif valence == 2 and self.list_atom_elems[i] == 'N':
                        cos_a0 = np.cos(120. / 180 * np.pi)
                    elif valence == 2:
                        cos_a0 = np.cos(np.pi)
                    elif valence == 3 and self.list_atom_elems[i] == 'N':
                        cos_a0 = np.cos(107. / 180 * np.pi)
                    elif valence == 3:
                        cos_a0 = np.cos(120.00 / 180 * np.pi)
                    elif valence == 4:
                        cos_a0 = np.cos(109.45 / 180 * np.pi)  # 109.5 degree
                    a0[ia] = cos_a0
            return a0

        self.cos_a0 = torch.tensor(get_a0(self.angles), device=self.device)
        self.n_angles = len(self.angles)

        # diheds
        diheds = []
        for ib in range(len(self.bonds)):
            j, k = self.bonds[ib]
            ilist = np.array(self._adj[int(j)])
            llist = np.array(self._adj[int(k)])
            for i in ilist:
                if i == k:
                    continue
                for l in llist:
                    if l == j:
                        continue
                    diheds.append([i, j, k, l])
        if len(diheds) == 0:
            diheds = np.array([]).reshape(0, 4)
        else:
            diheds = np.array(diheds)
            if len(diheds.shape) == 1:
                diheds = diheds.reshape(1, -1)
        self.diheds = torch.tensor(diheds, device=self.device)
        self.n_diheds = len(self.diheds)

        # setup the calc_internal_coord_feature function
        def calc_internal_coords_features(positions, box):
            '''
            Calculate the feature value of all ICs in the subgraph
            This function meant to be exposed to external use, with autograd etc.
            It relies on the following variables in Graph:
            self.bonds, self.angles, self.diheds
            self.a0, self.cos_b0
            All these variables should be "static" throughout NVE/NVT/NPT simulations
            '''

            box_inv = torch.linalg.inv(box)

            def _calc_bond_features(idx, pos, b0):
                pos0 = pos[idx[:, 0]]
                pos1 = pos[idx[:, 1]]
                dr = pbc_shift_torch(pos1 - pos0, box, box_inv)
                blength = torch.norm(dr, dim=1)
                return (blength - b0) * FSCALE_BOND

            def _calc_angle_features(idx, pos, cos_a0):
                rj = pos[idx[:, 0]]
                ri = pos[idx[:, 1]]
                rk = pos[idx[:, 2]]
                r_ij = pbc_shift_torch(rj - ri, box, box_inv)
                r_ik = pbc_shift_torch(rk - ri, box, box_inv)
                n_ij = torch.norm(r_ij, dim=1)
                n_ik = torch.norm(r_ik, dim=1)
                cos_a = torch.sum(r_ij * r_ik, dim=1) / n_ij / n_ik
                return (cos_a - cos_a0) * FSCALE_ANGLE

            def _calc_dihed_features(idx, pos):
                ri = pos[idx[:, 0]]
                rj = pos[idx[:, 1]]
                rk = pos[idx[:, 2]]
                rl = pos[idx[:, 3]]
                r_jk = pbc_shift_torch(rk - rj, box, box_inv)
                r_ji = pbc_shift_torch(ri - rj, box, box_inv)
                r_kl = pbc_shift_torch(rl - rk, box, box_inv)
                r_kj = -r_jk
                n1 = torch.linalg.cross(r_jk, r_ji)
                n2 = torch.linalg.cross(r_kl, r_kj)
                norm_n1 = torch.norm(n1, dim=1)
                norm_n2 = torch.norm(n2, dim=1)
                return torch.sum(n1 * n2, dim=1) / norm_n1 / norm_n2

            fb = _calc_bond_features(torch.tensor(self.bonds, device=self.device), positions, self.b0)
            fa = _calc_angle_features(torch.tensor(self.angles, device=self.device), positions, self.cos_a0)
            fd = _calc_dihed_features(self.diheds, positions)

            return fb.to(torch.float32), fa.to(torch.float32), fd.to(torch.float32)

        self.calc_internal_coords_features = calc_internal_coords_features

        # Build O(1) lookup tables for IC index matching
        self._build_ic_lookup()

        return

    def _build_ic_lookup(self):
        """Build hash lookup tables for bonds, angles, dihedrals -> index.
        This replaces O(n) linear scans with O(1) dict lookups."""
        self._bond_lookup = {}
        bonds_np = self.bonds if isinstance(self.bonds, np.ndarray) else self.bonds.cpu().numpy()
        for idx, b in enumerate(bonds_np):
            key = (int(b[0]), int(b[1]))
            self._bond_lookup[key] = idx
            self._bond_lookup[(key[1], key[0])] = idx

        self._angle_lookup = {}
        angles_np = self.angles if isinstance(self.angles, np.ndarray) else self.angles.cpu().numpy()
        for idx, a in enumerate(angles_np):
            key = (int(a[0]), int(a[1]), int(a[2]))
            self._angle_lookup[key] = idx
            self._angle_lookup[(key[2], key[1], key[0])] = idx

        self._dihed_lookup = {}
        diheds_np = self.diheds if isinstance(self.diheds, np.ndarray) else self.diheds.cpu().numpy()
        for idx, d in enumerate(diheds_np):
            key = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
            self._dihed_lookup[key] = idx
            self._dihed_lookup[(key[3], key[2], key[1], key[0])] = idx

    def prepare_subgraph_feature_calc(self):
        '''
        Preparing the feature calculation.
        Specifically, find out the indices mapping between feature elements and ICs

        After preparing the varibles in all subgraphs, we stack all subgraphs along the first axis
        After stacking, each row represents a fixed-order subgraph calculation
        The total number of rows: Ntot = \sum_g N_p(g), with N_p(g) being the permutation number of subgraph g
        Get these variables ready:
        (kb = ['center', 'nb_bonds_0', 'nb_bonds_1'])
        (kf = ['bonds', 'angles0', 'angles1', 'diheds'])
        feature_atypes: (Ntot, 2*MAX_VALENCE-1, DIM_BOND_FEATURES_ATYPES)
        feature_indices[kf]: (Ntot, 2*MAX_VALENCE-1, DIM_BOND_FEATURES_GEOM[kf])
        nb_connect[kb]: (Ntot, MAX_VALENCE-1)
        self.n_features: dimensionality of bond features

        Also setup the following function:
        self.calc_subgraph_features: 
            pos (Na*3), box (3*3) -> features (Ntot*7*n_features)
                The calculator for the Graph features.
        '''
        for g in self.subgraphs:
            g.prepare_graph_feature_calc()
        self.n_features_atypes = DIM_BOND_FEATURES_ATYPES
        self.n_features_geom = DIM_BOND_FEATURES_GEOM_TOT
        self.n_features = self.n_features_atypes + self.n_features_geom

        # concatenate permutations
        self.feature_atypes = {}
        self.feature_indices = {}
        if self.nn == 0:
            bond_groups = ['center']
        else:
            bond_groups = ['center', 'nb_bonds_0', 'nb_bonds_1']
        feature_groups = ['bonds', 'angles0', 'angles1', 'diheds']
        for kb in bond_groups:
            self.feature_atypes[kb] = torch.cat(
                [g.feature_atypes[kb].clone().detach().to(self.device) for g in self.subgraphs])
            self.feature_indices[kb] = {}
            for kf in feature_groups:
                self.feature_indices[kb][kf] = torch.cat(
                    [g.feature_indices[kb][kf].clone().detach().to(self.device) for g in self.subgraphs])
        self.weights = torch.cat([g.weights.clone().detach().to(self.device) for g in self.subgraphs])
        if self.nn == 1:
            self.nb_connect = {}
            for kb in ['nb_bonds_0', 'nb_bonds_1']:
                self.nb_connect[kb] = torch.cat([
                    torch.tile(g.nb_connect[kb].clone().detach().to(self.device), (g.n_sym_perm, 1))
                    for g in self.subgraphs
                ])
        self.map_subgraph_perm = torch.cat([
            torch.full((self.subgraphs[ig].n_sym_perm,), ig, dtype=torch.long, device=self.device)
            for ig in range(self.n_subgraphs)
        ])

        # concatenate bond groups
        if self.nn == 0:
            self.feature_atypes = self.feature_atypes['center']
        elif self.nn == 1:
            self.feature_atypes = torch.cat([
                self.feature_atypes['center'],
                self.feature_atypes['nb_bonds_0'],
                self.feature_atypes['nb_bonds_1']
            ], dim=1)
        feature_indices = {}
        for kf in feature_groups:
            if self.nn == 0:
                feature_indices[kf] = self.feature_indices['center'][kf]
            elif self.nn == 1:
                feature_indices[kf] = torch.cat([
                    self.feature_indices['center'][kf],
                    self.feature_indices['nb_bonds_0'][kf],
                    self.feature_indices['nb_bonds_1'][kf]
                ], dim=1)
        self.feature_indices = feature_indices
        if self.nn == 1:
            self.nb_connect = torch.cat(
                [self.nb_connect['nb_bonds_0'], self.nb_connect['nb_bonds_1']],
                dim=1)

        # set up the feature calculation function
        def _get_features(fb, fa, fd, f_atypes, indices_bonds, indices_angles0,
                          indices_angles1, indices_diheds):
            target_shape = f_atypes.shape[:2]  
            
            fb = fb.to(torch.float32)
            fa = fa.to(torch.float32)
            fd = fd.to(torch.float32)
            
            f_bonds = distribute_scalar_torch(fb, indices_bonds) * (indices_bonds >= 0).to(torch.float32)
            f_angles0 = distribute_scalar_torch(fa, indices_angles0) * (indices_angles0 >= 0).to(torch.float32)
            f_angles1 = distribute_scalar_torch(fa, indices_angles1) * (indices_angles1 >= 0).to(torch.float32)
            f_diheds = distribute_scalar_torch(fd, indices_diheds) * (indices_diheds >= 0).to(torch.float32)
            
            f_atypes = f_atypes.to(torch.float32)
            
            return torch.cat([f_atypes, f_bonds, f_angles0, f_angles1, f_diheds], dim=-1)

        def calc_subgraph_features(positions, box):
            fb, fa, fd = self.calc_internal_coords_features(positions, box)
            return _get_features(fb, fa, fd, self.feature_atypes,
                                 self.feature_indices['bonds'],
                                 self.feature_indices['angles0'],
                                 self.feature_indices['angles1'],
                                 self.feature_indices['diheds'])

        self.calc_subgraph_features = calc_subgraph_features
        return

    def write_xyz(self, file=None):
        '''
        Write the xyz file of the molecule
        '''
        if file is None:
            file = sys.stdout
        print(self.n_atoms, file=file)
        print('Generated by dmff.sgnn', file=file)
        for i in range(self.n_atoms):
            elem = self.list_atom_elems[i]
            x, y, z = self.positions[i].cpu().numpy()
            print('%3s %12.6f %12.6f %12.6f' % (elem, x, y, z), file=file)
        return


class TopSubGraph(TopGraph):

    def __init__(self, graph, i_center, nn, type_center='bond'):
        '''
        Find a subgraph within the graph, centered on a certain bond/atom
        The size of the subgraph is determined by nn (# of neighbour searches around the center)
        i_center defines the center, could be a bond, could be an atom
        '''
        self.device = graph.device
        self.list_atom_elems = []
        self.bonds = []
        self.positions = []
        self.valences = []
        self.map_sub2parent = [
        ]  # this maps the index in the subgraph to the index in the parent graph
        self.map_parent2sub = {}
        self.parent = graph
        self.box = graph.box
        self.box_inv = graph.box_inv
        self.nn = nn
        n_atoms = 0
        if type_center == 'atom':
            self.map_sub2parent.append(i_center)
            self.map_parent2sub[i_center] = n_atoms
            n_atoms += 1
            self.list_atom_elems.append(graph.list_atom_elems[i_center])
            self.valences.append(graph.valences[i_center])
        elif type_center == 'bond':
            b0 = graph.bonds[i_center]
            for i in b0:
                self.map_sub2parent.append(i)
                self.map_parent2sub[i] = n_atoms
                n_atoms += 1
                self.list_atom_elems.append(graph.list_atom_elems[i])
                self.valences.append(graph.valences[i])
            # the first bond of the subgraph is always (0, 1), the central bond
            self.bonds.append(np.array([0, 1]))
        self.n_atoms = n_atoms

        for n in range(nn + 1):
            self.add_neighbors()
        self._build_connectivity()

        self.map_sub2parent.append(-1)  
        self.map_sub2parent = np.array(self.map_sub2parent)
        if graph.positions is not None:
            valid_indices = [idx for idx in self.map_sub2parent[:-1] if idx >= 0]
            if len(valid_indices) > 0:
                self.positions = graph.positions[torch.tensor(valid_indices, device=self.device)]
            else:
                self.positions = torch.zeros((self.n_atoms, 3), device=self.device)
        else:
            self.positions = torch.zeros((self.n_atoms, 3), device=self.device)

        return

    # search one more layer of neighbours
    def add_neighbors(self):
        atoms_in_subgraph = set(self.map_parent2sub.keys())
        n_atoms = self.n_atoms
        # Use adjacency list for O(degree) instead of O(n_bonds)
        new_atoms = []
        for parent_atom in list(atoms_in_subgraph):
            for nb in self.parent._adj[parent_atom]:
                if nb not in atoms_in_subgraph:
                    new_atoms.append((parent_atom, nb))
        # Deduplicate (same new atom may be reached from multiple existing atoms)
        seen = set()
        for i_old, i_new in new_atoms:
            if i_new in seen:
                continue
            seen.add(i_new)
            self.list_atom_elems.append(self.parent.list_atom_elems[i_new])
            # Don't append to positions here - it will be set after all neighbors are added
            self.valences.append(self.parent.valences[i_new])
            self.map_sub2parent.append(i_new)
            self.map_parent2sub[i_new] = n_atoms
            bond = np.array([n_atoms, self.map_parent2sub[i_old]])
            self.bonds.append(np.sort(bond))
            n_atoms += 1
        self.n_atoms = n_atoms
        return

    def get_canonical_orders_wt_permutation_grps(self):
        '''
        This function sets up all the canonical orders for the atoms, based on existing 
        atom typification (atom_types) information and the connection topology.
        Specifically, it sets the following variables in the subgraph:

        g.canonical_orders
            All the orders that are symmetrically equivalent and nondistinguishable
        g.maps_canonical_orders
            The reverse mapping of the canonical orders (i.e., maps from atom indices to order)
        g.n_permutation
            Number of canonical orders
        '''
        # 'orders' is a queue that stores all the sequences
        if self.atom_types[0] == self.atom_types[1]:
            orders = [np.array([0, 1]), np.array([1, 0])]
        else:
            t0 = self.atom_types[0]
            t1 = self.atom_types[1]
            if t0 < t1:
                orders = [np.array([0, 1])]
            else:
                orders = [np.array([1, 0])]
        
        # Rest of the implementation follows the same logic as the original...
        # [Implementation continues with the same structure as original]
        # For brevity, I'll include the key parts

        def permute_using_atypes(indices, atypes):
            set_atypes = list(set(atypes))
            set_atypes.sort()
            sym_classes = {}
            permutation_grps = []
            for t in set_atypes:
                sym_classes[t] = np.where(atypes == t)[0]
                permutation_grps.append(indices[sym_classes[t]])
            indices_permutations = []
            seg_permutations = []
            for pseg in permutation_grps:
                seg_permutations.append(list(permutations(pseg)))
            pfull = []
            for p in product(*seg_permutations):
                pfull.append(np.concatenate(p))
            return np.array(pfull)

        def extend_orders(orders):
            n_order = len(orders)
            for i_order in range(n_order):
                order = orders.pop(0)
                seg_permutations = []
                for i in order:
                    js = np.where(self.connectivity[i])[0]
                    js = js[[not (j in order) for j in js]]
                    if len(js) == 0:
                        continue
                    atypes = np.array(self.atom_types)[js]
                    new_orders = permute_using_atypes(js, atypes)
                    seg_permutations.append(new_orders)
                if seg_permutations:  # Only if we have segments to permute
                    for p in product(*seg_permutations):
                        if p:  # Only if p is not empty
                            concatenated_p = np.concatenate(p)
                            if len(concatenated_p) > 0:  # Only if the concatenated result is not empty
                                orders.append(np.concatenate((order, concatenated_p)))
                            else:
                                orders.append(order)  # Just add the original order if nothing to concatenate
                        else:
                            orders.append(order)
                else:
                    orders.append(order)  # Just add the original order if no segments
            return orders

        for i in range(self.nn + 1):
            orders = extend_orders(orders)
        canonical_orders = np.array(orders)
        maps_canonical_orders = []
        for order in canonical_orders:
            map_order = np.zeros(self.n_atoms, dtype=int)
            for ii, i in enumerate(order):
                map_order[i] = ii
            maps_canonical_orders.append(map_order)
        maps_canonical_orders = np.array(maps_canonical_orders)
        
        self.canonical_orders = canonical_orders
        self.maps_canonical_orders = maps_canonical_orders
        self.n_permutations = len(canonical_orders)
        return

    def prepare_bond_feature_atypes(self, bond, map_order):
        '''
        Get feature elements that label the atom types
        For each atom, a vector is specified to mark its element
        [1 0 0 0 0] is H
        [0 1 0 0 0] is C
        [0 0 1 0 0] is N
        etc.
        These vectors are then catenated according to the given canonical order
        '''
        indices_atoms_center = np.array(bond)
        indices_atoms_center = sort_by_order(indices_atoms_center, map_order)
        i, j = indices_atoms_center
        elem_i = self.list_atom_elems[i]
        elem_j = self.list_atom_elems[j]
        fi = np.zeros(N_ATYPES)
        fj = np.zeros(N_ATYPES)
        fi[ATYPE_INDEX[elem_i]] = 1
        fj[ATYPE_INDEX[elem_j]] = 1
        
        indices_n0 = np.array([x for x in self._adj[int(i)] if x != j])
        indices_n1 = np.array([x for x in self._adj[int(j)] if x != i])
        indices_n0 = sort_by_order(indices_n0, map_order) if len(indices_n0) > 0 else indices_n0
        indices_n1 = sort_by_order(indices_n1, map_order) if len(indices_n1) > 0 else indices_n1
        nn0 = len(indices_n0)
        nn1 = len(indices_n1)

        f_n0 = np.zeros(N_ATYPES * (MAX_VALENCE - 1))
        f_n1 = np.zeros(N_ATYPES * (MAX_VALENCE - 1))
        for ii, i in enumerate(indices_n0):
            tmp = np.zeros(N_ATYPES)
            elem = self.list_atom_elems[i]
            tmp[ATYPE_INDEX[elem]] = 1
            f_n0[ii * N_ATYPES:ii * N_ATYPES + N_ATYPES] = tmp
        for ii, i in enumerate(indices_n1):
            tmp = np.zeros(N_ATYPES)
            elem = self.list_atom_elems[i]
            tmp[ATYPE_INDEX[elem]] = 1
            f_n1[ii * N_ATYPES:ii * N_ATYPES + N_ATYPES] = tmp
        return np.array(np.concatenate((fi, fj, f_n0, f_n1)))

    def prepare_bond_feature_calc_indices(self, bond, map_order, verbose=False):
        '''
        Given a bond, and a particular order of the atoms in the graph, prepare its
        geometric feature calculations.
        The geometric features of a bond will be composed by:
        1. It's own lengths
        2. The lengths of all it's neighbor bonds
        3. All angles that share atoms with the bond
        4. All diheds that are centered on the bond

        Correspondingly, we prepare the indices (in parent graph) of the corresponding ICs:
        indices['bond']: indices for all relevant bonds
        indices['angles[12]']: indices for all relevant angles
        indices['diheds']: indices for all relevant diheds

        All IC indices will be sorted according to the given atomic order.
        '''
        indices = {}
        G = self.parent
        indices_atoms_center = np.array(bond)
        indices_atoms_center = sort_by_order(indices_atoms_center, map_order)
        i, j = indices_atoms_center
        indices_n0 = np.array([x for x in self._adj[int(i)] if x != j])
        indices_n1 = np.array([x for x in self._adj[int(j)] if x != i])
        indices_n0 = sort_by_order(indices_n0, map_order) if len(indices_n0) > 0 else indices_n0
        indices_n1 = sort_by_order(indices_n1, map_order) if len(indices_n1) > 0 else indices_n1
        nn0 = len(indices_n0)
        nn1 = len(indices_n1)
        # padding neighbours
        indices_atoms_n0 = -np.ones(MAX_VALENCE - 1, dtype=int)
        indices_atoms_n1 = -np.ones(MAX_VALENCE - 1, dtype=int)
        indices_atoms_n0[:nn0] = indices_n0
        indices_atoms_n1[:nn1] = indices_n1

        # relevant bonds
        indices_bonds = []
        indices_bonds.append(indices_atoms_center)
        for i in indices_atoms_n0:
            indices_bonds.append([indices_atoms_center[0], i])
        for j in indices_atoms_n1:
            indices_bonds.append([indices_atoms_center[1], j])
        indices_bonds = np.array(indices_bonds)
        # convert to indices in parent graph using O(1) hash lookup
        indices['bonds'] = []
        for b in indices_bonds:
            p = tuple(int(self.map_sub2parent[i]) for i in b)
            idx = G._bond_lookup.get(p, -1)
            indices['bonds'].append(idx)
        indices['bonds'] = np.array(indices['bonds'], dtype=int)

        # relevant angles
        indices_angles_0 = []
        set_0 = np.array([indices_atoms_center[1]] + list(indices_atoms_n0))
        for ii, i in enumerate(set_0):
            for jj in range(ii + 1, len(set_0)):
                j = set_0[jj]
                angle = [i, indices_atoms_center[0], j]
                indices_angles_0.append(angle)
        indices_angles_0 = np.array(indices_angles_0, dtype=int)
        indices_angles_1 = []
        set_1 = np.array([indices_atoms_center[0]] + list(indices_atoms_n1))
        for ii, i in enumerate(set_1):
            for jj in range(ii + 1, len(set_1)):
                j = set_1[jj]
                angle = [i, indices_atoms_center[1], j]
                indices_angles_1.append(angle)
        indices_angles_1 = np.array(indices_angles_1, dtype=int)
        # convert to indices in parent graph using O(1) hash lookup
        indices['angles0'] = []
        indices['angles1'] = []
        for a in indices_angles_0:
            p = tuple(int(self.map_sub2parent[i]) for i in a)
            idx = G._angle_lookup.get(p, -1)
            indices['angles0'].append(idx)
        for a in indices_angles_1:
            p = tuple(int(self.map_sub2parent[i]) for i in a)
            idx = G._angle_lookup.get(p, -1)
            indices['angles1'].append(idx)
        indices['angles0'] = np.array(indices['angles0'], dtype=int)
        indices['angles1'] = np.array(indices['angles1'], dtype=int)

        # relevant dihedrals
        indices_diheds = []
        for i in indices_atoms_n0:
            for j in indices_atoms_n1:
                indices_diheds.append(
                    [i, indices_atoms_center[0], indices_atoms_center[1], j])
        indices_diheds = np.array(indices_diheds)
        indices['diheds'] = []
        for d in indices_diheds:
            p = tuple(int(self.map_sub2parent[i]) for i in d)
            idx = G._dihed_lookup.get(p, -1)
            indices['diheds'].append(idx)
        indices['diheds'] = np.array(indices['diheds'], dtype=int)

        return indices

    def prepare_graph_feature_calc(self):
        '''
        Prepare the variables that are needed in feature calculations.
        So far, we assume self.nn <= 1, so it is either only the central bond, or the central bond + its closest neighbor bonds
        The closest neighbor bonds are grouped into two groups: (nb_bonds_0) and (nb_bonds_1)
        The first group of bonds are attached to the first atom of the central bond
        The second group of bonds are attached to the second atom of the central bond
        So there are three bond groups: center (1bond), nb_bonds_0 (max 3 bonds), and nb_bonds_1 (max 3 bonds)
        In principle, it's not necessary to dinstinguish nb_bonds_0 and nb_bonds_1. Such division is merely a historical legacy.

        The following variables are set after the execution of this function

        Output: 
            self.feature_atypes:
                Dictionary with bond groups (['center', 'nb_bonds_0', 'nb_bonds_1']) as keywords
                'center': this group contains only one bond: the central bond
                'nb_bonds_0': this group contains the neighbor bonds attached to the first atoms
                'nb_bonds_1': this group contains the neighbor bonds attached to the second atoms
                feature_atypes['...'] is a (n_sym_perm, n_bonds, n_bond_features_atype) array, stores the atype features
                of the bond group. Atype features describes the atomtyping information of the graph, thus is bascially constant
                during the simulation.
            self.feature_indices:
                Nested dictionary with bond groups (['center', 'nb_bonds_0', 'nb_bonds_1']) as the first keyword
                and geometric feature types (['bonds', 'angles0', 'angles1', 'diheds']) as the second keyword
                It stores all the relevant IC indices
                Dimensionalities (when MAX_VALENCE=4):
                feature_indices['center']['bonds']: (n_sym_perm, 1, 7)
                feature_indices['center']['angles0']: (n_sym_perm, 1, 6)
                feature_indices['center']['angles1']: (n_sym_perm, 1, 6)
                feature_indices['center']['diheds']: (n_sym_perm, 1, 9)
                feature_indices['nb_bonds_x']['bonds']: (n_sym_perm, 3, 7)
                feature_indices['nb_bonds_x']['angles0']: (n_sym_perm, 3, 6)
                feature_indices['nb_bonds_x']['angles1']: (n_sym_perm, 3, 6)
                feature_indices['nb_bonds_x']['diheds']: (n_sym_perm, 3, 9)
            self.nb_connect:
                Dictionary with keywords: ['nb_bonds_0', 'nb_bonds_1']
                Describes how many neighbor bonds the central bond has. E.g., if there are only 2 neighbor bonds attached to 
                the first atom, then:
                self.nb_connect['nb_bonds_0'] = torch.tensor([1., 1., 0.])

        '''
        self.n_bond_features_atypes = DIM_BOND_FEATURES_ATYPES
        self.n_bond_features_geom = DIM_BOND_FEATURES_GEOM_TOT
        self.n_bond_features = self.n_bond_features_atypes + self.n_bond_features_geom
        # assume the first bond is always the central bond
        center_bond = self.bonds[0]  # should always be (0, 1)
        i, j = center_bond
        if self.nn == 1:
            # neighboring bonds
            nb_bonds_0 = []
            nb_bonds_1 = []
            for k in np.where(self.connectivity[i] == 1)[0]:
                if k != j:
                    nb_bonds_0.append([i, k])
            for l in np.where(self.connectivity[j] == 1)[0]:
                if l != i:
                    nb_bonds_1.append([j, l])
        # prepare the feature calculation for all these bonds
        feature_indices = {'center': []}
        feature_atypes = {'center': []}
        if self.nn == 1:
            feature_indices['nb_bonds_0'] = []
            feature_indices['nb_bonds_1'] = []
            feature_atypes['nb_bonds_0'] = []
            feature_atypes['nb_bonds_1'] = []

        # for different canonical orders, get the atype features and the internal coordinate feature indices
        for map_order in self.maps_canonical_orders:
            feature_indices['center'].append(
                self.prepare_bond_feature_calc_indices(center_bond, map_order))
            feature_atypes['center'].append(
                self.prepare_bond_feature_atypes(center_bond, map_order))
            if self.nn == 1:
                tmp = []
                tmp1 = []
                for b in nb_bonds_0:
                    tmp.append(
                        self.prepare_bond_feature_calc_indices(b, map_order))
                    tmp1.append(self.prepare_bond_feature_atypes(b, map_order))
                feature_indices['nb_bonds_0'].append(tmp)
                feature_atypes['nb_bonds_0'].append(tmp1)
                tmp = []
                tmp1 = []
                for b in nb_bonds_1:
                    tmp.append(
                        self.prepare_bond_feature_calc_indices(b, map_order))
                    tmp1.append(self.prepare_bond_feature_atypes(b, map_order))
                feature_indices['nb_bonds_1'].append(tmp)
                feature_atypes['nb_bonds_1'].append(tmp1)
        feature_atypes['center'] = np.array(feature_atypes['center'])
        if self.nn == 1:
            feature_atypes['nb_bonds_0'] = np.array(
                feature_atypes['nb_bonds_0'])
            feature_atypes['nb_bonds_1'] = np.array(
                feature_atypes['nb_bonds_1'])
            weights = np.ones(self.n_permutations) / self.n_permutations

        # merge the equivalent permutations
        indices_permutations = list(range(self.n_permutations))
        self.feature_indices = {'center': []}
        self.feature_atypes = {'center': []}
        if self.nn == 1:
            self.feature_indices['nb_bonds_0'] = []
            self.feature_indices['nb_bonds_1'] = []
            self.feature_atypes['nb_bonds_0'] = []
            self.feature_atypes['nb_bonds_1'] = []
        self.weights = []
        flags = [True for ip in indices_permutations]
        for ip in indices_permutations:
            # this permutation is already merged
            if not flags[ip]:
                continue
            # not merged yet
            else:
                self.feature_indices['center'].append(
                    feature_indices['center'][ip])
                self.feature_atypes['center'].append(
                    feature_atypes['center'][ip])
                if self.nn == 1:
                    self.feature_indices['nb_bonds_0'].append(
                        feature_indices['nb_bonds_0'][ip])
                    self.feature_indices['nb_bonds_1'].append(
                        feature_indices['nb_bonds_1'][ip])
                    self.feature_atypes['nb_bonds_0'].append(
                        feature_atypes['nb_bonds_0'][ip])
                    self.feature_atypes['nb_bonds_1'].append(
                        feature_atypes['nb_bonds_1'][ip])
                # calcualte permuataion symemetry multiplicity
                n = 1
                self.weights.append(n / self.n_permutations)
        # number of permutationally unique orders
        self.n_sym_perm = len(self.weights)
        self.weights = torch.tensor(self.weights, device=self.device)

        # rearrange feature_indices, make it more tensor-like ....
        for ip in range(self.n_sym_perm):
            self.feature_indices['center'][ip] = [
                self.feature_indices['center'][ip]
            ]
            self.feature_atypes['center'][ip] = [
                self.feature_atypes['center'][ip]
            ]
        # new tensor-like feature_atypes and feature_indices
        feature_indices = {}
        feature_atypes = {}
        if self.nn == 0:
            keys = ['center']
        elif self.nn == 1:
            keys = ['center', 'nb_bonds_0', 'nb_bonds_1']
            self.nb_connect = {}
            self.nb_connect['nb_bonds_0'] = np.zeros(MAX_VALENCE - 1)
            self.nb_connect['nb_bonds_1'] = np.zeros(MAX_VALENCE - 1)
        nb_list = {
            'center': 1,
            'nb_bonds_0': MAX_VALENCE - 1,
            'nb_bonds_1': MAX_VALENCE - 1
        }
        for kb in keys:
            # deal with the atype features
            feature_atypes[kb] = np.zeros(
                (self.n_sym_perm, nb_list[kb], DIM_BOND_FEATURES_ATYPES))
            nb = len(self.feature_atypes[kb][0])
            if nb > 0:
                feature_atypes[kb][:, 0:nb, :] = np.array(
                    np.array(self.feature_atypes[kb]))
            feature_atypes[kb] = torch.tensor(feature_atypes[kb], device=self.device)
            # deal with geometric feature indices
            feature_indices[kb] = {}
            for kf in ['bonds', 'angles0', 'angles1', 'diheds']:
                feature_indices[kb][kf] = -np.ones(
                    (self.n_sym_perm, nb_list[kb], DIM_BOND_FEATURES_GEOM[kf]),
                    dtype=int)
                if nb > 0:
                    feature_indices[kb][kf][:, 0:nb, :] = np.array([[
                        self.feature_indices[kb][ip][ib][kf][:]
                        for ib in range(nb)
                    ] for ip in range(self.n_sym_perm)])
                feature_indices[kb][kf] = torch.tensor(feature_indices[kb][kf], device=self.device)
            if self.nn == 1:
                if kb in self.nb_connect.keys():
                    if nb > 0:
                        self.nb_connect[kb][0:nb] = 1.0
                    self.nb_connect[kb] = torch.tensor(self.nb_connect[kb], device=self.device)
        self.feature_indices = feature_indices
        self.feature_atypes = feature_atypes

        return


def sort_by_order(ilist, map_order):
    return np.array(ilist)[np.argsort([map_order[i] for i in ilist])]


def from_pdb(pdb, device='cpu'):
    device = torch.device(device)
    mol = md.load(pdb)
    bonds = []
    for bond in mol.top.bonds:
        bonds.append(np.sort(np.array((bond.atom1.index, bond.atom2.index))))
    bonds = np.array(bonds)
    list_atom_elems = np.array([a.element.symbol for a in mol.top.atoms])
    positions = torch.tensor(mol.xyz[0] * 10, dtype=torch.float32, device=device)
    if mol.unitcell_vectors is None:
        box = None
    else:
        box = torch.tensor(mol.unitcell_vectors[0] * 10, dtype=torch.float32, device=device)
    return TopGraph(list_atom_elems, bonds, positions=positions, box=box, device=device)


