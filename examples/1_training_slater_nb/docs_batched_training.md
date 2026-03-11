# Batched Training for DMFF Backend

## Overview

This documentation explains the implementation of batched accelerated training in `train_dimer_batched.py`. The goal of this new script is to accelerate the training of OpenMM DMFF (Differentiable Molecular Force Field) potentials by allowing the JAX/XLA compiler to evaluate multiple, distinct molecular topologies in parallel, essentially achieving the performance benefits of "padding different molecules" without the overhead and rigidity of traditional zero-padding arrays.

## Traditional Padding vs. Dictionary Unrolling

In standard machine learning frameworks (like PyTorch or generic JAX networks), processing different graphs/molecules in parallel usually requires pad-and-mask strategies. For example, you might create a tensor `(Batch_Size, Max_Atoms, 3)` and use dummy zeros for smaller molecules.

However, in the **OpenMM DMFF ecosystem**, potentials (`createPotential`) are intrinsically bound to their underlying topology and atom typing during initialization. They compile into JAX functions that expect an exact structural map (neighbor lists, parameter references). You cannot dynamically pass padded `atom_types` arrays to these potentials.

### The Solution: JAX Dictionary Batching (Loop Unrolling)

Instead of manual array padding, `train_dimer_batched.py` achieves batched parallelism by taking advantage of JAX's `jax.jit` and XLA (Accelerated Linear Algebra) loop unrolling capabilities over dictionaries.

In the training step, instead of doing an independent `train_step` (and an SGD optimizer update) for each molecule pair individually, we:

1.  **Gather a Global Batch Data Dictionary**: Sample one random frame-batch for *every* molecular pair.
2.  **Unroll the Loss Evaluation**: Provide the dictionary to a single `@jit` compiled `batched_loss` function. By iterating over a fixed static list of keys (`dimer_train`), JAX unrolls the loop during tracing.
3.  **Fuse into a Single XLA Graph**: The XLA compiler automatically aggregates the 10+ different DMFF potential evaluations into one giant computational graph, computing their forces in parallel across the GPU cores.
4.  **Single Optimizer Step**: We sum their losses and calculate gradients with respect to the `params` globally, performing one large, stable optimizer step.

This is **mathematically and computationally superior** to traditional padding because:
- **No Wasted FLOPs**: We do not perform meaningless calculations on padded "dummy" atoms.
- **Perfect Stability**: All parameters receive uniform gradient updates per epoch, eliminating the noise of sequential pair-by-pair stochastic updates.
- **Topology Flexibility**: Different molecules retain their exact OpenMM DMFF optimized neighbor lists and functions.

## Implementation Details

### The Unified Batch Loss Function

```python
@jit
def batched_loss(params_local, global_batch_data):
    total_loss = jnp.float32(0.0)
    pair_losses = {}
    
    # Iterate over static keys: JAX unrolls this into a single parallel graph
    for pair_key in dimer_train:
        scan_data = global_batch_data[pair_key]
        ...
        pred = jnp.stack([e_ex, e_es, e_pol, e_disp, e_dhf, e_tot], axis=0)
        errs = jnp.sum((ref - pred) ** 2 * (weights_pts[None, :] / norm), axis=1)
        pair_loss = jnp.sum(COMPONENT_WEIGHTS * errs)
        
        pair_losses[pair_key] = pair_loss
        total_loss += pair_loss
        
    return total_loss / len(dimer_train), pair_losses

# has_aux=True allows returning individual losses for logging 
# without disrupting the gradient calculation of the total scalar loss.
batched_loss_grad = jit(value_and_grad(batched_loss, has_aux=True))
```

### Execution

To run the new accelerated batch training script:

```bash
# Uses the same YAML configuration standard as the original backend
python3 train_dimer_batched.py --config train_config.yaml
```

*Note: The initial JIT compilation will take slightly longer (~1 minute) because XLA is fusing multiple distinct OpenMM topologies into a single execution kernel. Once compiled, epochs will run significantly faster and gradients will be universally stable.*

## Extending Component Weights

In `train_dimer_batched.py`, the `COMPONENT_WEIGHTS` are currently set to `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` by default to avoid the previous issue where `pol` and `disp` trends were sacrificed to fit the overarching `tot` curve. If you wish to optimize specific components based on their energy scale, consider altering this array directly or moving it to `train_config.yaml`.
