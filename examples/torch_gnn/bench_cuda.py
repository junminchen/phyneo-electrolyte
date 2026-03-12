"""Standalone CUDA benchmark for MolGNNForce forward pass."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from graph_torch import from_pdb
from gnn_torch import MolGNNForce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BOX_SIZE = 80.0
print(f"Device: {device}")

pdb_file = os.path.join(os.path.dirname(__file__), 'model_15000.pdb')
print(f"Loading PDB: {pdb_file}")

# Build graph and model
G = from_pdb(pdb_file, device=device)
if G.box is None:
    box = torch.tensor(
        [[BOX_SIZE, 0, 0], [0, BOX_SIZE, 0], [0, 0, BOX_SIZE]],
        dtype=torch.float32,
        device=device,
    )
    G.set_box(box)
    print(f"No box in PDB; using cubic box with side length {BOX_SIZE:.1f} A")
print(f"Atoms: {G.n_atoms}, Bonds: {len(G.bonds)}")

model = MolGNNForce(G=G, nn=1, device=device)

# Load params from existing pickle (JAX format -> convert)
import pickle
params_file = os.path.join(os.path.dirname(__file__), '..', '3_training_sgnn_bonding', 'params_sgnn.pickle')
#params_file = None
if os.path.exists(params_file):
    print(f"Loading params from: {params_file}")
    with open(params_file, 'rb') as f:
        params = pickle.load(f)
    # Convert JAX arrays to torch tensors
    for k in params:
        if hasattr(params[k], '__array__'):
            params[k] = torch.tensor(np.array(params[k]), dtype=torch.float32)
        elif isinstance(params[k], list):
            params[k] = [torch.tensor(np.array(x), dtype=torch.float32) if hasattr(x, '__array__') else x for x in params[k]]

    # Try loading - shapes must match
    try:
        from gnn_torch import prm_transform_f2i
        params_internal = prm_transform_f2i(params, model.n_layers)
        with torch.no_grad():
            model.w = params_internal['w'].to(device)
            for i, layer in enumerate(model.fc0_layers):
                w = params_internal['fc0.weight'][i].to(device)
                b = params_internal['fc0.bias'][i].to(device)
                if w.shape == layer.weight.shape:
                    layer.weight.copy_(w)
                    layer.bias.copy_(b)
            for i, layer in enumerate(model.fc1_layers):
                w = params_internal['fc1.weight'][i].to(device)
                b = params_internal['fc1.bias'][i].to(device)
                if w.shape == layer.weight.shape:
                    layer.weight.copy_(w)
                    layer.bias.copy_(b)
            fw = params_internal['fc_final.weight'].to(device)
            fb = params_internal['fc_final.bias'].to(device)
            if fw.shape == model.fc_final.weight.shape:
                model.fc_final.weight.copy_(fw)
                model.fc_final.bias.copy_(fb)
        print("Params loaded successfully")
    except Exception as e:
        print(f"Param load failed (using random): {e}")
else:
    print("No params file found, using random weights")

model.eval()
model.to_device(device)

positions = G.positions.clone().to(device)
box = G.box.clone().to(device) if G.box is not None else None

# Warmup
print("\nWarmup (5 iterations)...")
for _ in range(5):
    with torch.no_grad():
        e = model.forward(positions, box)
torch.cuda.synchronize()
print(f"  Energy = {e.item():.6f}")

# Benchmark forward pass only
n_iters = 100
print(f"\nBenchmark: {n_iters} forward passes...")
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(n_iters):
    with torch.no_grad():
        e = model.forward(positions, box)
torch.cuda.synchronize()
t1 = time.perf_counter()

total = t1 - t0
per_iter = total / n_iters * 1000  # ms
print(f"  Total: {total:.3f} s")
print(f"  Per forward: {per_iter:.3f} ms")
print(f"  Throughput: {n_iters/total:.1f} forward/s")

# Benchmark with gradient (energy + forces)
print(f"\nBenchmark: {n_iters} forward+backward (forces)...")
positions_grad = positions.clone().requires_grad_(True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(n_iters):
    e = model.forward(positions_grad, box)
    forces = torch.autograd.grad(e, positions_grad, retain_graph=False)[0]
torch.cuda.synchronize()
t1 = time.perf_counter()

total = t1 - t0
per_iter = total / n_iters * 1000
print(f"  Total: {total:.3f} s")
print(f"  Per forward+backward: {per_iter:.3f} ms")
print(f"  Throughput: {n_iters/total:.1f} steps/s")
print(f"  Force shape: {forces.shape}")
