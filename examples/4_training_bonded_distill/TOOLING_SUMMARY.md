# Bonded Parameter Tooling Summary

## Context

The current goal is not to build another full intramolecular ML potential for small-molecule electrolytes. The more practical target is a bonded parameter workflow that can produce:

- bond terms
- angle terms
- dihedral / torsion terms

for use in OpenMM and DMFF.

This is especially relevant for:

- common solvent molecules such as EC, DMC, EMC, DEC, and FEC
- charged small molecules and organic ions
- ABn or high-symmetry anions such as PF6, BF4, and DFP

## High-Level Conclusion

For liquid small-molecule electrolytes, a dedicated bonded fitting or bonded distillation workflow is more attractive than carrying a full sGNN intramolecular model into production MD.

The main reasons are:

- lower runtime cost
- easier export into classical MD engines
- easier debugging and inspection
- more direct integration with `MPIDForce`

The most useful outcome is therefore a bonded parameter model or workflow, not a new general-purpose graph neural network.

## What Should Be Learned

The target should be a bonded model that covers:

- bond equilibrium values and force constants
- angle equilibrium values and force constants
- torsion coefficients and periodicities

The most important point is that torsions should be treated as a first-class target. For electrolyte molecules, the main failure of generic force fields is often not bond stretching itself but torsional energetics and conformational preferences.

## Distillation Strategy

If a strong teacher model such as ByteFF is available, the preferred route is:

- use ByteFF as teacher
- fit a classical bonded student model
- export bonded parameters for OpenMM or DMFF

The fitting target should be teacher observables, not raw teacher parameter values.

Preferred teacher targets:

- optimized geometries
- local energies
- local forces
- selected local curvatures or Hessian-derived information
- 1D torsion scans

This is preferable to directly learning teacher parameter values because bonded parameters are often non-identifiable. Different parameter combinations can reproduce very similar observables.

## Tooling Recommendations

### 1. ByteFF / ByteFF-Pol

Best role:

- teacher model for bonded distillation
- source of high-quality bonded and nonbonded priors

Strengths:

- already accessible through the local API
- predicts bonded and nonbonded terms together
- appropriate as a strong reference for solvent molecules and ions

Limits:

- not necessarily the cleanest ready-made OpenMM bonded production pipeline
- direct parameter copying is less attractive than observable matching

Use recommendation:

- use ByteFF as the initial teacher for the bonded distillation workflow

References:

- ByteFF paper: <https://pubs.rsc.org/en/content/articlehtml/2025/sc/d4sc06640e>
- ByteFF-Pol preprint: <https://arxiv.org/abs/2508.08575>

### 2. OpenFF Sage + BespokeFit

Best role:

- open-source route for small organic molecules and organic ions
- especially useful for torsion refits

Strengths:

- mature open-source tooling
- strong support for torsion fitting workflows
- good fit for new organic ions and flexible small molecules

Limits:

- less obviously suitable as a direct answer for PF6, BF4, and similar anions
- still may require bespoke handling for unusual charged species

Use recommendation:

- strong reference design for how to implement torsion-focused fitting
- good fallback for organic molecules and organic ions

References:

- OpenFF force fields: <https://openforcefield.org/force-fields/force-fields/>
- BespokeFit docs: <https://docs.openforcefield.org/projects/bespokefit/en/latest/>
- Bespoke workflows: <https://docs.openforcefield.org/projects/bespokefit/en/0.3.0/users/bespoke-workflows.html>

### 3. QUBEKit

Best role:

- high-quality bespoke parameter derivation for new molecules and new ions

Strengths:

- QM-driven parameter derivation
- better suited than generic rule-based assignment when a molecule is new or unusual

Limits:

- more involved than quick generator-style tools
- less convenient when the goal is just to get a fast baseline

Use recommendation:

- strong option for difficult new ions when higher setup cost is acceptable

Reference:

- QUBEKit repository: <https://github.com/qubekit/QUBEKit>

### 4. GAFF2 / AmberTools

Best role:

- baseline bonded generator for general organic molecules and many charged organics

Strengths:

- widely used
- practical baseline
- accessible from existing OpenMM tooling

Limits:

- not the preferred final answer for special ions or highly sensitive torsions

Use recommendation:

- use as a baseline or initializer, not as the final high-confidence answer for difficult anions

References:

- OpenMM forcefields helpers: <https://github.com/openmm/openmmforcefields>
- AmberTools Antechamber: <https://ambermd.org/antechamber/ac.html>

### 5. LigParGen / OPLS

Best role:

- quick baseline parameter generation

Strengths:

- easy starting point
- useful when a fast initial parameter set is needed

Limits:

- not the strongest route for difficult torsions or unusual ions
- better as a prior than as a final high-confidence bonded model

Use recommendation:

- use as initialization or fallback
- avoid treating it as the best available route for special electrolyte ions

References:

- LigParGen OpenMM tutorial: <https://zarbi.chem.yale.edu/ligpargen/openMM_tutorial.html>
- SEAMM LigParGen notes: <https://molssi-seamm.github.io/forcefield_step/user_guide/ligpargen/index.html>

## Practical Recommendation for This Repository

The recommended strategy is:

1. Build a dedicated bonded distillation workflow under `examples/4_training_bonded_distill/`.
2. Use ByteFF as the initial teacher.
3. Fit a classical bonded student model covering bond, angle, and dihedral terms.
4. Treat torsion fitting as a required part of the workflow, not an optional extra.
5. Export the fitted parameters into OpenMM- and DMFF-friendly forms.

## Recommended Use by Molecule Class

### Standard Solvents

Examples:

- EC
- DMC
- EMC
- DEC
- FEC

Recommendation:

- start from OPLS or GAFF2 if needed
- refine with ByteFF-guided bonded distillation if torsions or local stiffness are not satisfactory

### Organic Ions

Recommendation:

- OpenFF Sage + BespokeFit is a strong open-source reference path
- ByteFF-guided distillation is also attractive if the API already covers the chemistry

### PF6, BF4, DFP, and Similar ABn Ions

Recommendation:

- do not rely blindly on automatic generic bonded assignment
- prefer bespoke fitting or ByteFF-guided bonded distillation

These species are precisely where generic transferable bonded models are most likely to underperform.

## Final Decision

The main engineering target should be:

- a bonded parameterization workflow

not:

- another deployment-scale intramolecular ML force

The most promising first implementation is:

- ByteFF teacher
- classical bonded student
- explicit support for bond, angle, and torsion fitting
- export to OpenMM and DMFF
