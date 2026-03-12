# Bonded Distillation PRD

## Goal

Build a dedicated bonded fitting workflow for small-molecule electrolytes that can learn or distill bonded terms from a higher-level teacher model and export parameters usable in OpenMM and DMFF.

The first target is a practical replacement for the current "use OPLS bonded unless something is obviously wrong" workflow. The new workflow should support:

- bond stretching
- angle bending
- dihedral / torsion fitting

The initial teacher will be the ByteFF API because it is already available in the local environment. The student model will be a classical bonded force field with explicit bond, angle, and torsion parameters.

## Motivation

The current sGNN route is expressive, but for liquid small-molecule electrolytes it is often more expensive than necessary. In many cases the main missing physics is not a fully learned intramolecular potential, but a better bonded model for:

- equilibrium geometries
- local stiffness
- torsional profiles
- intramolecular response under coordination

For this problem class, a distilled bonded model has several advantages:

- much cheaper than sGNN during MD
- easier to combine with MPIDForce
- easier to inspect and debug
- easier to export into standard OpenMM force terms

## Users

- researchers building liquid electrolyte force fields
- developers who already have ByteFF or QM access and want a cheaper MD-ready bonded model
- users who want to fit only a subset of molecules such as PF6, BF4, DFP, EC, DMC, or EMC

## Non-Goals

- replacing MPID or the pairwise ML nonbonded model
- learning a universal graph neural network for bonded energies
- supporting arbitrary reactive bond breaking or topology changes
- solving condensed-phase parameterization end-to-end in the first version

## Product Scope

### In Scope for V1

- create a bonded distillation dataset from teacher calls
- fit bond and angle terms from geometry, force, and local curvature information
- fit dihedral terms from 1D torsion scans
- export fitted parameters into an OpenMM-compatible representation
- support per-molecule or per-residue fitting
- support initializing missing terms from OPLS or another prior parameter set
- support fitting only selected molecules while leaving the rest on existing bonded parameters

### Out of Scope for V1

- automated QM job generation
- condensed-phase reweighting
- full Hessian matching for large molecules by default
- simultaneous fitting of nonbonded and bonded terms
- a GUI or notebook-first workflow

## Core Product Idea

Use a teacher-student workflow:

- teacher:
  - ByteFF API in the first implementation
  - later optionally QM references
- student:
  - explicit classical bonded terms
  - bond equilibrium values and force constants
  - angle equilibrium values and force constants
  - torsion Fourier coefficients or another chosen torsion form

The fitting target should be teacher observables, not just teacher parameter values. This avoids parameter non-identifiability becoming the main optimization target.

## Special Case: Teacher API Returns Parameters Only

The initial external integration may expose ByteFF only as an API of the form:

- input: `SMILES`
- output: bonded parameter tables

In that case, the workflow still remains viable, but it must support two distinct distillation modes.

### Mode A: Parameter Distillation

Directly learn the mapping from molecular structure to bonded parameters.

This should not be implemented as a single variable-length whole-molecule regression target. Instead, the workflow should decompose the output into local training samples:

- one sample per bond term
- one sample per angle term
- one sample per dihedral term

Each sample should use the local chemical environment as input.

Examples:

- bond sample input:
  - the two endpoint atoms
  - bond order or bond class
  - local neighbor environments
  - ring and conjugation features
  - formal charge or related local charge descriptors
- angle sample input:
  - the three atoms defining the angle
  - local substituent environments
  - ring and aromaticity information
- dihedral sample input:
  - the four atoms defining the torsion
  - the central bond class
  - local substituent environments
  - ring membership
  - conjugation and symmetry descriptors

The outputs should be:

- bond:
  - equilibrium value
  - force constant
- angle:
  - equilibrium value
  - force constant
- dihedral:
  - periodic torsion coefficients
  - phase and periodicity when required by the target form

This mode is useful for building a reusable parameter generator, especially when the teacher API can be queried at scale.

### Mode B: Parameter-to-Observable Distillation

If the API returns only parameters, the workflow should still be able to turn those parameters into teacher observables.

Recommended flow:

1. query the teacher API for bonded parameters
2. build a teacher bonded force from those parameters
3. generate local displaced conformations and torsion scan conformations
4. evaluate teacher energies and optionally forces
5. fit the student parameters against these teacher observables

This mode is more robust than pure parameter matching because it constrains the behavior induced by the teacher parameters, not only the raw numbers themselves.

### Recommended Strategy

The recommended implementation is a two-stage workflow:

1. parameter imitation pretraining
   - query the teacher API for many molecules
   - train local predictors for bond, angle, and dihedral parameters
2. observable-level refinement
   - build teacher bonded energies from the returned parameter tables
   - refine the student against local energies, forces, and torsion scans

This strategy is preferable because:

- it uses the API efficiently
- it reduces sensitivity to parameter non-identifiability
- it gives dihedral fitting a physically meaningful target

### Technical Requirements for the Parameter-Only API Case

- canonicalize molecules before querying or storing results
- preserve atom mapping between returned parameter tables and internal graph indices
- support variable numbers of bond, angle, and dihedral terms across molecules
- support explicit symmetry handling for equivalent terms
- support torsion scan generation even when the teacher API itself does not directly return scan energies

### Special Attention for ABn and Symmetric Ions

Species such as PF6, BF4, and DFP require additional symmetry-aware handling.

Requirements:

- equivalent bonded terms should be grouped explicitly
- symmetry-equivalent bonds, angles, and dihedrals should be regularized or constrained together
- the workflow should detect when a teacher parameter table already encodes symmetry and preserve it during distillation

The API-only case is therefore still useful, but it reinforces the need for a symmetry-aware local bonded workflow rather than a naive whole-molecule parameter regression.

## Teacher Targets

The system should support collecting some or all of the following teacher data:

- optimized geometry
- energies for perturbed structures
- forces for perturbed structures
- selected Hessian blocks or projected local curvatures
- 1D torsion scan energies
- optionally 2D coupled torsion scan energies in a later phase

## Student Parameterization Targets

### Bond Terms

Fit:

- equilibrium bond length
- bond force constant

Sources:

- optimized geometry
- finite-displacement energies or forces
- optional Hessian-derived constraints

### Angle Terms

Fit:

- equilibrium angle
- angle force constant

Sources:

- optimized geometry
- finite-displacement energies or forces
- optional Hessian-derived constraints

### Dihedral Terms

Fit:

- periodic torsion coefficients
- phase and periodicity if required by target functional form

Sources:

- teacher torsion scans
- optionally local force matching around scan geometries

## Why Dihedrals Must Be First-Class

For electrolyte molecules, the main failure mode of generic bonded force fields is often not bond stretching. The largest chemically relevant errors frequently come from:

- carbonate torsions
- coupled internal rotations
- anion internal distortions
- environment-sensitive conformational preferences

This means the workflow must treat dihedral fitting as a primary deliverable, not a later add-on.

## Target Molecule Classes

### Standard Small Molecules

- EC
- DMC
- EMC
- DEC
- FEC

### ABn / High-Symmetry Anions

- PF6
- BF4
- DFP

These anions should be supported explicitly because they are common failure points for generic bonded force fields and are already important in the repository.

## User Stories

1. As a developer, I want to point the workflow at a PDB or topology plus a teacher API so I can generate a distilled bonded dataset.
2. As a developer, I want to fit only PF6 torsions and angle terms while leaving all solvent molecules unchanged.
3. As a researcher, I want to initialize unknown parameters from OPLS and refine them against teacher energies.
4. As a developer, I want to export fitted parameters into a form I can load into OpenMM or DMFF without hand-editing XML.
5. As a researcher, I want a report showing which bonded terms improved and which still have large residuals.

## Proposed Directory Contents

The new workflow will live under:

`examples/4_training_bonded_distill/`

Planned files:

- `PRD.md`
- `README.md`
- `generate_teacher_data.py`
- `fit_bonded.py`
- `fit_dihedrals.py`
- `export_openmm.py`
- `plot_fit_report.py`
- `configs/`
- `outputs/`

## Proposed Data Flow

1. Load molecule topology and coordinates.
2. Query the teacher for reference observables.
3. Build a distilled dataset:
   - equilibrium structures
   - displaced structures
   - torsion scan structures
4. Optimize bonded parameters against those targets.
5. Export the fitted parameter set.
6. Validate against held-out scans or local perturbations.

## Functional Requirements

### Dataset Generation

- accept input structures from PDB and simple coordinate files
- accept molecule identity or residue name selection
- generate perturbations around equilibrium geometry
- generate torsion scan coordinates for selected rotatable bonds
- cache teacher outputs to disk

### Fitting

- support fitting a subset of terms while freezing others
- support separate weights for geometry, energy, force, Hessian, and scan losses
- support initialization from ByteFF, OPLS, or user-supplied parameters
- support per-molecule fitting and batched multi-molecule fitting
- support restart from saved optimizer state

### Export

- export to a JSON or pickle intermediate format
- export to OpenMM XML-compatible bonded parameter records
- export to a DMFF-friendly parameter bundle

### Reporting

- summarize train and validation losses
- plot torsion scan overlays
- report parameter deltas from initialization
- highlight poorly fit terms

## Loss Design

The objective should be modular. A first version can use:

`L_total = w_geom * L_geom + w_energy * L_energy + w_force * L_force + w_scan * L_scan + w_reg * L_reg`

Possible additions later:

- `L_hessian`
- `L_freq`
- `L_condensed_phase_proxy`

### Suggested First-Pass Definitions

- `L_geom`: error in optimized internal coordinates
- `L_energy`: relative energy error over local perturbations
- `L_force`: force-matching error on perturbed geometries
- `L_scan`: relative energy error on torsion scan grids
- `L_reg`: regularization toward initial bonded parameters

## Architecture Requirements

### Teacher Layer

Provide a thin adapter interface so the fitting code does not depend directly on ByteFF-specific calls.

Required teacher operations:

- optimize geometry or return a reference geometry
- evaluate energy
- evaluate forces
- generate or evaluate torsion scans

### Student Layer

Represent bonded parameters explicitly and independently from the teacher.

The student must be able to:

- compute bonded energy for one molecule
- compute gradients with respect to bonded parameters
- handle frozen and trainable parameter masks

### Optimization Layer

- use JAX where practical because the repository already depends on it
- allow a pure NumPy or SciPy fallback if some export paths are easier outside JAX
- keep the parameterization deterministic and restartable

## Validation Requirements

The workflow is considered useful only if it can demonstrate:

- lower torsion scan error than the starting OPLS-like baseline
- stable optimized geometry reproduction
- reasonable transfer to unseen conformers for the same molecule
- successful export into OpenMM bonded terms

## Risks

### Parameter Non-Identifiability

Different bonded parameters can match similar local observables. Mitigation:

- fit observables, not only parameter values
- regularize toward a prior
- fit subsets of terms in stages

### Teacher Bias

If ByteFF has systematic errors, distillation will preserve them. Mitigation:

- keep teacher adapters pluggable
- support later replacement with QM data

### Torsion Coupling

Independent 1D scans may miss coupled torsions. Mitigation:

- start with 1D scans
- add optional 2D scans for known coupled systems later

### Overfitting to Gas-Phase Data

A bonded model fitted only to isolated molecules may not reflect coordinated states. Mitigation:

- include teacher data from ion-coordinated cluster geometries where needed

## Milestones

### Milestone 1

- create folder structure
- define teacher adapter interface
- generate equilibrium and local perturbation datasets
- fit bond and angle terms for one solvent molecule

### Milestone 2

- add torsion scan generation
- fit dihedral parameters for one solvent molecule
- export OpenMM-ready bonded parameters

### Milestone 3

- support PF6, BF4, and DFP
- support mixed initialization from ByteFF and OPLS
- generate comparison plots and fit reports

### Milestone 4

- connect exported bonded terms into the MD workflow
- compare `MPID + bonded-distilled` against `MPID + OPLS-bonded`

## Success Metrics

- a fitted bonded model can be produced for at least one solvent and one ABn anion
- torsion scan RMSE is materially below the starting OPLS baseline
- exported parameters can be loaded into OpenMM without manual patching
- the workflow is scriptable and reproducible from the command line

## Immediate Next Step

Implement the smallest viable version:

- one molecule at a time
- ByteFF teacher only
- bond, angle, and 1D dihedral support
- OpenMM export for the fitted bonded terms
