"""Candidate molecule list for bonded distillation and replay sampling.

This file intentionally keeps a single top-level dictionary:

    MOLECULE_SMILES[name] -> smiles

The list mixes:
- core replay molecules already close to the current chemistry space
- boron-containing expansion molecules
- silicon-containing expansion molecules
- aluminum-containing expansion molecules

The goal is not to be exhaustive. It is a practical seed set for:
- querying an external parameter API
- building a replay set to avoid forgetting old chemistry
- expanding bonded coverage for new element families
"""

MOLECULE_SMILES = {
    # Core replay / electrolyte solvents and related ions
    "ethylene_carbonate": "O=C1OCCO1",
    "propylene_carbonate": "CC1COC(=O)O1",
    "dimethyl_carbonate": "COC(=O)OC",
    "ethyl_methyl_carbonate": "CCOC(=O)OC",
    "diethyl_carbonate": "CCOC(=O)OCC",
    "fluoroethylene_carbonate": "O=C1OC(F)CO1",
    "1_2_dimethoxyethane": "COCCOC",
    "diglyme": "COCCOCCOC",
    "triglyme": "COCCOCCOCCOC",
    "1_3_dioxolane": "C1COCO1",
    "1_4_dioxane": "O1CCOCC1",
    "tetrahydrofuran": "C1CCOC1",
    "acetonitrile": "CC#N",
    "dimethyl_sulfoxide": "CS(=O)C",
    "sulfolane": "O=S1(=O)CCCC1",
    "bis_fluorosulfonyl_imide": "[N-](S(=O)(=O)F)S(=O)(=O)F",
    "bis_trifluoromethanesulfonyl_imide": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "hexafluorophosphate": "F[P-](F)(F)(F)(F)F",
    "tetrafluoroborate": "[B-](F)(F)(F)F",
    "difluorophosphate": "[O-]P(=O)(F)F",

    # Boron family expansion
    "boron_trifluoride": "FB(F)F",
    "boric_acid": "B(O)(O)O",
    "trimethyl_borate": "B(OC)(OC)OC",
    "triethyl_borate": "B(OCC)(OCC)OCC",
    "tripropyl_borate": "B(OCCC)(OCCC)OCCC",
    "tributyl_borate": "B(OCCCC)(OCCCC)OCCCC",
    "triphenyl_borate": "B(Oc1ccccc1)(Oc1ccccc1)Oc1ccccc1",
    "tetramethoxyborate": "[B-](OC)(OC)(OC)OC",
    "tetraethoxyborate": "[B-](OCC)(OCC)(OCC)OCC",
    "tetrahydroxyborate": "[B-](O)(O)(O)O",

    # Silicon family expansion
    "tetramethylsilane": "C[Si](C)(C)C",
    "tetraethyl_orthosilicate": "CCO[Si](OCC)(OCC)OCC",
    "tetramethyl_orthosilicate": "CO[Si](OC)(OC)OC",
    "hexamethyldisiloxane": "C[Si](C)(C)O[Si](C)(C)C",
    "octamethyltrisiloxane": "C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C",
    "trimethylchlorosilane": "C[Si](C)(C)Cl",
    "trimethylfluorosilane": "C[Si](C)(C)F",
    "methyltrimethoxysilane": "C[Si](OC)(OC)OC",
    "ethyltriethoxysilane": "CC[Si](OCC)(OCC)OCC",
    "vinyltrimethoxysilane": "C=C[Si](OC)(OC)OC",

    # Aluminum family expansion
    "aluminum_trichloride": "Cl[Al](Cl)Cl",
    "aluminum_trifluoride": "F[Al](F)F",
    "tetrachloroaluminate": "[Al-](Cl)(Cl)(Cl)Cl",
    "hexafluoroaluminate": "[Al-](F)(F)(F)(F)(F)F",
    "trimethoxyaluminum": "CO[Al](OC)OC",
    "triethoxyaluminum": "CCO[Al](OCC)OCC",
    "triisopropoxyaluminum": "CC(C)O[Al](OC(C)C)OC(C)C",
}

