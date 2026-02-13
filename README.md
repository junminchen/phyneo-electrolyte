# PhyNEO-Electrolyte

A Hybrid Physics-Driven Neural Network Force Field for Liquid Electrolytes

[![arXiv](https://img.shields.io/badge/arXiv-2511.13294-b31b1b.svg)](https://arxiv.org/abs/2511.13294)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

PhyNEO-Electrolyte is a hybrid machine learning framework designed for the simulation of liquid electrolytes. It combines physics-based long-range interactions with neural network corrections for short-range and bonding behaviors.

This repository provides the implementation of:
- **Embedded Atom Pairwise Neural Network (EAPNN)**: For short-range pairwise potential corrections.
- **Sub-graph Neural Network (sGNN)**: For capturing bonding interactions.
- **Slater-Type Function Fitting**: For modeling short-range repulsion based on SAPT(DFT) EDA data.

Paper: [PhyNEO-Electrolyte: A Hybrid Physics-Driven Neural Network Force Field for Liquid Electrolytes](https://arxiv.org/abs/2511.13294)

## Features

- 🔬 **Hybrid Architecture**: Combines traditional physics models (long-range) with deep learning (short-range/bonding).
- ⚡ **JAX-Powered**: High-performance training and inference using JAX and Flax.
- 🧪 **Physical Consistency**: Enforces energy and force consistency throughout the training loop.

## Project Structure

```text
phyneo-electrolyte/
├── phyneo/                      # Core package
│   ├── models/                  
│   │   └── eapnn.py             # EAPNN and sGNN model definitions
│   └── utils/                   
│       ├── data_utils.py        # JAX/Torch dataset and plotting scripts
│       └── md_utils.py          # Molecular Dynamics utilities
├── examples/                    # Training and MD examples
│   ├── 1_training_slater_nb/    # Slater-type orbital fitting
│   ├── 2_training_pairwise_ml_nb/ # EAPNN training loop
│   ├── 3_training_sgnn_bonding/   # sGNN and total model training
│   └── md_simulation/           # MD simulation examples using dmff/openmm
├── data/                        # Training datasets (pickle format)
└── README.md
```

## Installation

```bash
pip install git+https://github.com/junminchen/DMFF.git@devel
git clone https://github.com/junminchen/phyneo-electrolyte.git
cd phyneo-electrolyte
git lfs pull
pip install -r requirements.txt
source set_pythonpath
```

## Quick Start

### 1. Short-range Reconstruction (EAPNN)
Refer to `examples/2_training_pairwise_ml_nb/train_eapnn.py`. This script includes:
- JAX-based training loop with early stopping.
- Validation on energy and forces.
- Real-time plotting of training progress.

### 2. Bonding Energy Correction (sGNN)
Refer to `examples/3_training_sgnn_bonding/train_total.py` for training the bonding terms.

## Data and Code Availability

### Data Availability
A subset of the short-range Slater-type function fitting data, sGNN and pairwise ML correction training sets is publicly released alongside the source code at [https://github.com/junminchen/phyneo-electrolyte](https://github.com/junminchen/phyneo-electrolyte).

### Code Availability
The code for PhyNEO-Electrolyte including the training scheme, trained model and MD examples are available at [https://github.com/junminchen/phyneo-electrolyte](https://github.com/junminchen/phyneo-electrolyte). 
The PhyNEO long-range parameters developed as previous methods is provided at [https://github.com/junminchen/PhyNEO](https://github.com/junminchen/PhyNEO). 
The underlying DMFF package is available at [https://github.com/junminchen/DMFF](https://github.com/junminchen/DMFF).

## Citation

If you use this work, please cite:

```bibtex
@article{chen2025hybrid,
  title={A Hybrid Physics-Driven Neural Network Force Field for Liquid Electrolytes},
  author={Chen, Junmin and Gao, Qian and Lin, Yange and Huang, Miaofei and Cheng, Zheng and Feng, Wei and Huang, Jianxing and Wang, Bo and Yu, Kuang},
  journal={arXiv preprint arXiv:2511.13294},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
