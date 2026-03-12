"""
Benchmark: sGNNForceFast in OpenMM via CallbackPyForce TorchForce.
"""
import warnings

warnings.filterwarnings("ignore")

import os
import sys
import time
import torch

import openmm as mm
from openmm import app, unit
from openmm.app import PDBFile, ForceField, Simulation
from CallbackPyForce import Callable, TorchForce as CallbackTorchForce

sys.path.insert(0, os.path.dirname(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)

from graph_torch import from_pdb, FSCALE_BOND, FSCALE_ANGLE, MAX_VALENCE
from gnn_torch import MolGNNForce
from sgnn_fast import sGNNForceFast, load_params_from_pickle

PDB_FILE = os.path.join(os.path.dirname(__file__), "model_15000.pdb")
PARAMS_FILE = os.path.join(REPO_DIR, "examples/md_simulation/params_sgnn.pickle")
FF_XML = os.path.join(os.path.dirname(__file__), "opls_solvent.xml")
BOX_SIZE = 80.0
DEVICE = "cuda:0"

print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")


class GraphAdapter:
    def __init__(self, G):
        self.bonds = G.bonds
        self.b0 = G.b0
        self.fscale_bond = FSCALE_BOND
        self.angles = G.angles
        self.cos_a0 = G.cos_a0
        self.fscale_angle = FSCALE_ANGLE
        self.diheds = G.diheds
        self.feature_atypes = G.feature_atypes
        self.feature_indices = G.feature_indices
        self.nb_connect = G.nb_connect
        self.weights = G.weights
        self.n_features = int(G.n_features)
        self.max_valence = MAX_VALENCE


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
    box = torch.tensor(
        [[BOX_SIZE, 0, 0], [0, BOX_SIZE, 0], [0, 0, BOX_SIZE]],
        dtype=torch.float32,
    )
    G = from_pdb(PDB_FILE)
    G.set_box(box)
    _ = MolGNNForce(G=G, nn=1)
    print(
        f"   {G.n_atoms} atoms, {len(G.bonds)} bonds, {len(G.subgraphs)} subgraphs "
        f"in {time.time()-t0:.1f}s"
    )

    print("2. Creating sGNNForceFast model...")
    model = sGNNForceFast(GraphAdapter(G), n_layers=(3, 2), sizes=[(40, 20, 20), (20, 10)])
    load_params_from_pickle(model, PARAMS_FILE)
    model.eval()
    model = model.to(DEVICE)

    with torch.no_grad():
        e = model(G.positions.clone().to(DEVICE), box.to(DEVICE))
    print(f"   Energy: {e.item():.2f} kcal/mol")

    print("3. Standalone PyTorch fwd+bwd...")
    pg = G.positions.clone().to(DEVICE).requires_grad_(True)
    bx = box.to(DEVICE)
    for _ in range(10):
        e = model(pg, bx)
        torch.autograd.grad(e, pg)[0]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        e = model(pg, bx)
        torch.autograd.grad(e, pg)[0]
    torch.cuda.synchronize()
    dt_torch = (time.perf_counter() - t0) / 200
    print(f"   {dt_torch*1000:.2f} ms/step")

    print("4. Creating CallbackPyForce wrapper...")
    callback_model = CallbackWrapper(model, box.to(DEVICE), DEVICE)
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
