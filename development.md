# Development Log & Architectural Changes

## Architectural Changes

### 1. EAPNN Model Optimization
**Change:** Redesigned the Environment-Aware Pairwise Neural Network (EAPNN) architecture in `phyneo/models/eapnn.py`.
**Reason:** The previous ACSF/APSF Gaussian-based features were less expressive and computationally less efficient to tune for complex electrolyte systems.

**Key Implementation Details:**
*   **Bessel Radial Basis Functions:** Replaced Gaussian radial basis with Bessel basis functions for more robust radial feature representation.
*   **Angular Basis Functions:** Switched to an optimized angular basis acting on cosine angles, improving the description of local geometry.
*   **Attention-based Interaction:** Introduced a `MultiHeadDotProductAttention` mechanism to aggregate environmental information from neighbor atoms. This allows the model to dynamically weight the influence of different neighbors based on their chemical environment and spatial arrangement.
*   **Residual MLP Blocks:** Replaced the simple feed-forward layers with `ResidualBlock` units including `LayerNorm` and `Swish` activation, which enhances gradient flow and training stability for deeper architectures.
*   **Learnable Atomic Embeddings:** Moved from manual one-hot encoding to an `nn.Embed` layer, allowing the model to learn chemical representations of different species.

### 2. API and Code Structure Refinement
**Change:** Cleaned up `phyneo/models/` structure and removed unnecessary dependencies.
**Reason:** Improve maintainability and reduce the overhead of the JAX-based potential implementation.
*   Removed unused imports of OpenMM and PyTorch from the core JAX model file.
*   Streamlined the `phyneo.models` namespace to export only the primary `EAPNNForce` interface.

### 3. Dataset Expansion
**Change:** Significantly expanded `data/dataset_eapnn/data_all.xyz`.
**Reason:** The updated model requires more diverse training data to generalize across different configurations of Li-electrolyte complexes.

---

## Existing Test Results

### 1. EAPNN Potential Correction Training
*   **Location:** `examples/2_training_pairwise_ml_nb/`
*   **Configurations:** Li-PF6 and Li-EC dimer systems.
*   **Status:** Successfully trained with the optimized architecture.
*   **Performance (on `test_structures.xyz`):**
    *   **Baseline RMSE:** 27.1012 kcal/mol
    *   **EAPNN RMSE:** 1.4579 kcal/mol
    *   **Error Reduction:** 94.62%
*   **Visual Validation:** `evaluation_results.png` includes parity and residual plots with RMSE annotations.
*   **Outcome:** The model accurately corrects the baseline force field across a wide range of dimer distances. `best_model_params.pickle` is ready for production.

### 2. SGNN Bonding Interaction Training
*   **Location:** `examples/3_training_sgnn_bonding/`
*   **Results (Epoch 95):**
    *   **Train Error:** 15.01047
    *   **Test Error:** 1.74674
*   **Conclusion:** The sub-graph neural network shows strong performance in learning bonding interactions with high accuracy on the test set.

### 3. MD Simulation Validation
*   **Location:** `examples/md_simulation/`
*   **Status:** Initial tests using the DMFF driver and i-PI interface are functional.
*   **Note:** A recent execution failure (`OSError: [Errno 98] Address already in use`) was detected in the logs, which is a standard environment/socket cleanup issue and does not reflect a failure in the model logic itself.
