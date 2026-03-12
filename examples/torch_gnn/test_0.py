import warnings
warnings.filterwarnings("ignore")

import os
import torch as pt
# from openmmtorch import TorchForce
import torch
import torch.nn as nn
import torch.nn.functional as F
import openmmtools
import pickle

# run simulations
import sys
from openmm.unit import *
from openmm.app import *
from openmm import *
from CallbackPyForce import Callable, TorchForce
from gnn_torch import MolGNNForce
from graph_torch import *
import graph_torch
import time



# === OPLS-LJ处理 ===
def OPLS_LJ(system):
    """应用OPLS-LJ组合规则"""
    forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
    nonbonded_force = forces.get('NonbondedForce')
    
    if nonbonded_force is None:
        raise ValueError("系统中未找到非键相互作用")
    
    lorentz = CustomNonbondedForce('4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)')
    lorentz.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    lorentz.addPerParticleParameter('sigma')
    lorentz.addPerParticleParameter('epsilon')
    lorentz.setCutoffDistance(nonbonded_force.getCutoffDistance())
    system.addForce(lorentz)
    
    LJset = {}
    for index in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
        LJset[index] = (sigma, epsilon)
        lorentz.addParticle([sigma, epsilon])
        nonbonded_force.setParticleParameters(index, charge, sigma, epsilon * 0)
    
    for i in range(nonbonded_force.getNumExceptions()):
        p1, p2, q, sig, eps = nonbonded_force.getExceptionParameters(i)
        lorentz.addExclusion(p1, p2)
        if eps._value != 0.0:
            sig14 = sqrt(LJset[p1][0] * LJset[p2][0])
            eps14 = sqrt(LJset[p1][1] * LJset[p2][1])
            nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)
    
    return system


# device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# pdb_file = 'peg4.pdb'
# pdb_file = 'model.pdb'
pdb_file = 'model_15000.pdb'

G = from_pdb(pdb_file, device=device)
model = MolGNNForce(G=G, nn=1)
model.load_params("params_sgnn_torch.pickle")
model.eval()
model.to_device(device)

cb = Callable(id(model), Callable.RETURN_ENERGY)
torch_force = TorchForce(cb)

# module = torch.jit.script(model)
# module.save(model_name)


# 创建系统
mol = PDBFile(pdb_file)
ff = ForceField('opls_solvent.xml')
system = ff.createSystem(
    mol.topology, 
    nonbondedMethod=PME, 
    nonbondedCutoff=1.0*nanometer,
    constraints=None, 
    rigidWater=True,
    removeCMMotion=True
)
system = OPLS_LJ(system)

system.removeForce(0)
system.removeForce(0)
system.removeForce(2)

system.addForce(torch_force)            # Add the TorchForce to your System

print(f"Num of forces: {system.getNumForces()}")

for item in system.getForces():
    print(item)

# 设置积分器和平台
integrator = LangevinIntegrator(298*kelvin, 1.0/picoseconds, 1.0*femtosecond)

platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}

# 创建模拟
simulation = Simulation(mol.topology, system, integrator, platform, properties)
simulation.context.setPositions(mol.positions)


# 设置输出
output_pdb = 'output.pdb'
steps = 10000
simulation.reporters.append(PDBReporter(output_pdb, 2000))
simulation.reporters.append(StateDataReporter(sys.stdout, 2000, step=True, potentialEnergy=True, temperature=True, speed=True))

# 运行模拟
start = time.time()
print(f"start: {start} 开始分子动力学模拟 ({steps} 步)...")
simulation.step(steps)
print(f"end: {time.time()-start:.2f}模拟完成，轨迹已保存至: {output_pdb}")
# print(f"Steps to simulate: {*(10**6)/1}")


