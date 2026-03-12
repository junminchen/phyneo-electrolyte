"""
Benchmark: MolGNNForce in OpenMM via CallbackPyForce TorchForce.
"""
import warnings

warnings.filterwarnings("ignore")

import os
import sys
import time
import pickle
import torch
import numpy as np
import openmm as mm
from openmm import app, unit
from openmm.app import PDBFile, ForceField, Simulation
from CallbackPyForce import Callable, TorchForce as CallbackTorchForce

sys.path.insert(0, os.path.dirname(__file__))

from graph_torch import from_pdb
from gnn_torch import MolGNNForce, prm_transform_f2i

PDB_FILE = os.path.join(os.path.dirname(__file__), "model_15000.pdb")
PARAMS_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "3_training_sgnn_bonding",
    "params_sgnn.pickle",
)
FF_XML = os.path.join(os.path.dirname(__file__), "opls_solvent.xml")
BOX_SIZE = 80.0
DEVICE = "cuda:0"

print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")


class CallbackWrapper:
    """Convert OpenMM nm/kJ units to the model's A/kcal units."""

    def __init__(self, core, box_angstrom, device):
        self.core = core
        self.device = torch.device(device)
        self.box_angstrom = box_angstrom.to(self.device)

    def __call__(self, positions_nm):
        positions_nm = positions_nm.to(device=self.device, dtype=torch.float32)
        energy_kcal = self.core(positions_nm * 10.0, self.box_angstrom)
        return energy_kcal * 4.184


def load_molgnn_params(model, params_file, device):
    with open(params_file, "rb") as f:
        params = pickle.load(f)

    for k in params:
        if hasattr(params[k], "__array__"):
            params[k] = torch.tensor(np.array(params[k]), dtype=torch.float32)
        elif isinstance(params[k], list):
            params[k] = [
                torch.tensor(np.array(x), dtype=torch.float32) if hasattr(x, "__array__") else x
                for x in params[k]
            ]

    params_internal = prm_transform_f2i(params, model.n_layers)
    with torch.no_grad():
        model.w = params_internal["w"].to(device)
        for i, layer in enumerate(model.fc0_layers):
            w = params_internal["fc0.weight"][i].to(device)
            b = params_internal["fc0.bias"][i].to(device)
            if w.shape == layer.weight.shape:
                layer.weight.copy_(w)
                layer.bias.copy_(b)
        for i, layer in enumerate(model.fc1_layers):
            w = params_internal["fc1.weight"][i].to(device)
            b = params_internal["fc1.bias"][i].to(device)
            if w.shape == layer.weight.shape:
                layer.weight.copy_(w)
                layer.bias.copy_(b)
        fw = params_internal["fc_final.weight"].to(device)
        fb = params_internal["fc_final.bias"].to(device)
        if fw.shape == model.fc_final.weight.shape:
            model.fc_final.weight.copy_(fw)
            model.fc_final.bias.copy_(fb)


def OPLS_LJ(system):
    from math import sqrt

    forces = {
        system.getForce(i).__class__.__name__: system.getForce(i)
        for i in range(system.getNumForces())
    }
    nb = forces["NonbondedForce"]
    lorentz = mm.CustomNonbondedForce(
        "4*epsilon*((sigma/r)^12-(sigma/r)^6);"
        "sigma=sqrt(sigma1*sigma2);epsilon=sqrt(epsilon1*epsilon2)"
    )
    lorentz.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    lorentz.addPerParticleParameter("sigma")
    lorentz.addPerParticleParameter("epsilon")
    lorentz.setCutoffDistance(nb.getCutoffDistance())
    system.addForce(lorentz)
    ljset = {}
    for idx in range(nb.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(idx)
        ljset[idx] = (sig, eps)
        lorentz.addParticle([sig, eps])
        nb.setParticleParameters(idx, q, sig, eps * 0)
    for i in range(nb.getNumExceptions()):
        p1, p2, q, sig, eps = nb.getExceptionParameters(i)
        lorentz.addExclusion(p1, p2)
        if eps._value != 0.0:
            sig14 = sqrt(ljset[p1][0]._value * ljset[p2][0]._value) * sig.unit
            eps14 = sqrt(ljset[p1][1]._value * ljset[p2][1]._value) * eps.unit
            nb.setExceptionParameters(i, p1, p2, q, sig14, eps)
    return system


def run_benchmark(warmup=500, bench=2000):
    print("\n1. Building graph (gnn_torch)...")
    t0 = time.time()
    G = from_pdb(PDB_FILE, device=DEVICE)
    box = torch.tensor(
        [[BOX_SIZE, 0, 0], [0, BOX_SIZE, 0], [0, 0, BOX_SIZE]],
        dtype=torch.float32,
        device=DEVICE,
    )
    G.set_box(box)
    print(f"   {G.n_atoms} atoms, {len(G.bonds)} bonds in {time.time()-t0:.1f}s")

    print("2. Creating MolGNNForce model...")
    model = MolGNNForce(G=G, nn=1, device=DEVICE)
    load_molgnn_params(model, PARAMS_FILE, DEVICE)
    model.eval()
    model.to_device(DEVICE)

    positions = G.positions.clone().to(DEVICE)
    with torch.no_grad():
        e = model.forward(positions, box)
    print(f"   Energy: {e.item():.2f} kcal/mol")

    print("3. Standalone PyTorch fwd+bwd...")
    positions_grad = positions.clone().requires_grad_(True)
    for _ in range(10):
        e = model.forward(positions_grad, box)
        torch.autograd.grad(e, positions_grad)[0]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        e = model.forward(positions_grad, box)
        torch.autograd.grad(e, positions_grad)[0]
    torch.cuda.synchronize()
    dt_torch = (time.perf_counter() - t0) / 100
    print(f"   {dt_torch*1000:.2f} ms/step")

    print("4. Creating CallbackPyForce wrapper...")
    callback_model = CallbackWrapper(model, box, DEVICE)
    cb = Callable(id(callback_model), Callable.RETURN_ENERGY)

    print("5. Creating OpenMM system...")
    pdb = PDBFile(PDB_FILE)
    box_nm_val = BOX_SIZE / 10.0
    pdb.topology.setPeriodicBoxVectors(
        [
            mm.Vec3(box_nm_val, 0, 0) * unit.nanometer,
            mm.Vec3(0, box_nm_val, 0) * unit.nanometer,
            mm.Vec3(0, 0, box_nm_val) * unit.nanometer,
        ]
    )
    ff = ForceField(FF_XML)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=True,
        removeCMMotion=True,
    )
    system = OPLS_LJ(system)

    to_rm = [
        i
        for i in range(system.getNumForces())
        if system.getForce(i).__class__.__name__
        in ("HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce")
    ]
    for i in sorted(to_rm, reverse=True):
        system.removeForce(i)

    callback_force = CallbackTorchForce(cb)
    callback_force.setForceGroup(1)
    system.addForce(callback_force)

    forces = [system.getForce(i).__class__.__name__ for i in range(system.getNumForces())]
    print(f"   Forces: {forces}")

    print("6. Running OpenMM simulation...")
    integrator = mm.LangevinIntegrator(
        298 * unit.kelvin,
        1.0 / unit.picoseconds,
        1.0 * unit.femtosecond,
    )
    platform = mm.Platform.getPlatformByName("CUDA")
    sim = Simulation(pdb.topology, system, integrator, platform, {"CudaPrecision": "mixed"})
    sim.context.setPositions(pdb.positions)

    state = sim.context.getState(getEnergy=True)
    print(f"   Initial energy: {state.getPotentialEnergy()}")

    print(f"   Warmup: {warmup} steps...")
    sim.step(warmup)

    print(f"   Benchmark: {bench} steps...")
    t0 = time.perf_counter()
    sim.step(bench)
    t1 = time.perf_counter()
    dt = t1 - t0
    ms_step = dt / bench * 1000
    ns_day = bench * 1e-6 / dt * 86400

    print(f"\n{'='*50}")
    print(f"RESULTS: {G.n_atoms} atoms")
    print(f"{'='*50}")
    print(f"  PyTorch fwd+bwd:  {dt_torch*1000:.2f} ms")
    print(f"  OpenMM step:      {ms_step:.2f} ms")
    print(f"  Performance:      {ns_day:.2f} ns/day")
    print(f"  Target:           >= 15 ns/day (for 10k atoms)")
    if ns_day >= 15:
        print("  Status:           PASSED")
    else:
        est_10k = ns_day * 15000 / 10000
        status = "PASSED" if est_10k >= 15 else "NEEDS WORK"
        print(f"  Estimated 10k:    ~{est_10k:.1f} ns/day {status}")


if __name__ == "__main__":
    run_benchmark()
