"""Charge-conservation checks independent of the normalization algorithm."""

from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((Path(__file__).with_name("charge_neutrality_cases.json")).read_text())
IONS = {"LiA": 1, "NaA": 1, "BF4": -1, "BOB": -1, "DFO": -1,
        "DFP": -1, "FSI": -1, "PF6": -1, "TFS": -1}


def parameters(name):
    root = ET.parse(ROOT/name).getroot()
    section, attr = ("ADMPPmeForce", "c0") if root.find("ADMPPmeForce") is not None else ("NonbondedForce", "charge")
    q = {}
    for atom in root.findall(section + "/Atom"):
        key, value = atom.get("type"), Decimal(atom.get(attr))
        if key in q and q[key] != value:
            raise AssertionError("Different monopoles for the same type/local-frame alternatives")
        q[key] = value
    residues = {r.get("name"): {a.get("name"): q[a.get("type")] for a in r.findall("Atom")}
                for r in root.findall("Residues/Residue")}
    return root, q, residues


class ChargeConservation(unittest.TestCase):
    def assert_charge(self, actual, expected=0, tolerance="1e-15"):
        self.assertLessEqual(abs(actual-Decimal(expected)), Decimal(tolerance))

    def test_formal_molecular_charges_and_polymer_ends(self):
        for name in CASES["forcefields"]:
            with self.subTest(xml=name):
                _, _, residues = parameters(name)
                for residue, atoms in residues.items():
                    if "TEO" in residues and residue in ("TER", "TEO"):
                        continue
                    self.assert_charge(sum(atoms.values()), IONS.get(residue, 0))
                if "TEO" in residues:
                    self.assert_charge(sum(residues["TER"].values()) + sum(residues["TEO"].values()))
                    self.assert_charge(2*sum(residues["TEO"].values()))

    def test_damping_charge_copies_match_every_monopole(self):
        for name in CASES["forcefields"]:
            with self.subTest(xml=name):
                root, q, _ = parameters(name)
                if root.find("ADMPPmeForce") is None:
                    continue
                for force in ("SlaterSrEsForce", "QqTtDampingForce"):
                    for atom in root.findall(force + "/Atom"):
                        self.assertEqual(Decimal(atom.get("Q")), q[atom.get("type")])

    def test_shared_electrostatics_agree(self):
        reference = ET.parse(ROOT/CASES["forcefields"][0]).getroot().find("ADMPPmeForce")
        expected = ET.tostring(reference)
        for name in CASES["forcefields"]:
            with self.subTest(xml=name):
                actual = ET.parse(ROOT/name).getroot().find("ADMPPmeForce")
                self.assertEqual(ET.tostring(actual), expected)

    def test_transfer_to_multiple_chain_lengths(self):
        for name in CASES["forcefields"]:
            _, _, residues = parameters(name)
            if "IN1" in residues:
                ends, repeat = 2*sum(residues["TE1"].values()), sum(residues["IN1"].values())
            elif "INT" in residues:
                ends = sum(residues["TER"].values()) + sum(residues.get("TEO", residues["TER"]).values())
                repeat = sum(residues["INT"].values())
            else:
                continue
            for n in (0, 1, 2, 4, 6, 14, 31, 99, 1000):
                with self.subTest(xml=name, internal_units=n):
                    self.assert_charge(ends+n*repeat, tolerance="1e-14")
                    if "TEO" in residues:
                        self.assert_charge(2*sum(residues["TEO"].values())+n*repeat, tolerance="1e-14")

    def test_pdb_composition_and_formal_charges(self):
        for case in CASES["pdbs"]:
            with self.subTest(**case):
                root, _, residues = parameters(case["xml"])
                elements = {a.get("name"): a.get("element") for a in root.findall("AtomTypes/Type")}
                expected = {r.get("name"): Counter(elements[a.get("type")] for a in r.findall("Atom"))
                            for r in root.findall("Residues/Residue")}
                groups = {}
                for line in (ROOT/case["pdb"]).read_text().splitlines():
                    if line.startswith(("ATOM  ", "HETATM")):
                        key = (line[21:27], line[17:20].strip())
                        atom = line[12:16].strip()
                        element = (line[76:78].strip() if len(line) >= 78 else "")
                        if not element:
                            element = atom[:2] if atom[:2] in ("Li", "Na") else atom[0]
                        groups.setdefault(key, Counter())[element.title()] += 1
                self.assertTrue(groups)
                total = Decimal(0)
                for (_, residue), composition in groups.items():
                    self.assertEqual(composition, expected[residue])
                    total += sum(residues[residue].values())
                self.assert_charge(total, case.get("charge", 0), tolerance="1e-12")


if __name__ == "__main__":
    unittest.main()
