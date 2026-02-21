import os
import torch
import torch.nn as nn
from openmm import openmm
try:
    import openmmtorch
except ImportError:
    print("Warning: openmmtorch is not installed. ML forces cannot be added to OpenMM.")

class EAPNNForceWrapper(nn.Module):
    """
    Wraps the EAPNNForceTorch model to hold static topology variables as buffers.
    This allows the model to accept only (positions, box) from OpenMM TorchForce.
    """
    def __init__(self, core_model, pairs, valid_mask, topo_nblist, topo_mask, mol_ID, atype_indices):
        super().__init__()
        self.core_model = core_model
        # Register static topology as non-trainable buffers
        self.register_buffer("pairs", torch.tensor(pairs, dtype=torch.long))
        self.register_buffer("valid_mask", torch.tensor(valid_mask, dtype=torch.bool))
        self.register_buffer("topo_nblist", torch.tensor(topo_nblist, dtype=torch.long))
        self.register_buffer("topo_mask", torch.tensor(topo_mask, dtype=torch.bool))
        self.register_buffer("mol_ID", torch.tensor(mol_ID, dtype=torch.long))
        self.register_buffer("atype_indices", torch.tensor(atype_indices, dtype=torch.long))
        
        # openmm-torch convention: standard positions are in nanometers (nm)
        # However, ML formulations are usually in Angstroms (A). We apply the conversion factor.
        self.nm_to_angstrom = 10.0
        # Energy is returned by ML in kcal/mol (usually). OpenMM expects kJ/mol.
        self.kcal_to_kj = 4.184

    def forward(self, positions, boxvectors):
        # OpenMM TorchForce provides `positions` mapping to (N, 3) in nm.
        pos_A = positions * self.nm_to_angstrom
        box_A = boxvectors * self.nm_to_angstrom
        
        energy_kcal = self.core_model(
            pos_A, box_A, 
            self.pairs, self.valid_mask, 
            self.topo_nblist, self.topo_mask, 
            self.mol_ID, self.atype_indices
        )
        # Return scalar energy scaled to kJ/mol
        return energy_kcal * self.kcal_to_kj

class sGNNForceWrapper(nn.Module):
    """
    Wraps the sGNNForceTorch model.
    """
    def __init__(self, core_model):
        super().__init__()
        self.core_model = core_model
        self.nm_to_angstrom = 10.0
        self.kcal_to_kj = 4.184

    def forward(self, positions, boxvectors):
        pos_A = positions * self.nm_to_angstrom
        box_A = boxvectors * self.nm_to_angstrom
        
        energy_kcal = self.core_model(pos_A, box_A)
        return energy_kcal * self.kcal_to_kj

def create_and_add_torch_force(system, wrapped_model, force_group=1):
    """
    Compiles the wrapped PyTorch module into TorchScript and adds it to the OpenMM System.
    """
    import tempfile
    
    # Trace or Script the model
    # TorchScript is strict; scripting is often better for dynamic control flow, but 
    # tracing works perfectly if sizes are strictly static. We will use scripting.
    scripted_model = torch.jit.script(wrapped_model)
    
    # Save temporarily to load into OpenMM-Torch
    fd, temp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    
    try:
        scripted_model.save(temp_path)
        
        # Instantiate OpenMM TorchForce
        torch_force = openmmtorch.TorchForce(temp_path)
        
        # PBC is required since our ml models calculate distances using `box`
        torch_force.setUsesPeriodicBoundaryConditions(True)
        torch_force.setForceGroup(force_group)
        
        system.addForce(torch_force)
        print(f"Added ML TorchForce (Group {force_group}) to OpenMM System.")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return system
