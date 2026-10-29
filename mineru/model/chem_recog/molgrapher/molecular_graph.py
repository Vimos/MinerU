"""Molecular-graph -> RDKit molecule, with MolGrapher's post-processing.

Port of ``molgrapher/utils/utils_graph.py::MolecularGraph`` (nodes-only
variants) and of the parts of ``molgrapher/utils/utils_postprocessing.py`` that
are reachable for the ``gc_gcn_model`` path:

* aromatic bond outside a ring -> SINGLE  (``postprocess_aromatic_bonds``)
* ring whose majority bond type is aromatic -> all bonds aromatic
  (``postprocess_aromatic_rings``)
* sanitize with ``SANITIZE_KEKULIZE`` only, exactly like upstream
  (``try_sanitize_molecule``)
* one round of isolated-atom removal + recursion (``remove_isolated_atoms``)
* last resort ``Chem.MolFromSmiles("C")`` (-> confidence 0 upstream)

The PaddleOCR abbreviation *detector* is not used: ``gc_gcn_model`` predicts
abbreviation groups directly as atom classes (vocabulary_atoms_141.json
contains ``CF3``/``OMe``/``Ph``/...).  Those classes are expanded here with the
MolDepictor single-attachment-point SMILES table
(``data/abbreviations.json``, ``*`` = attachment atom); an abbreviation without
a usable entry becomes a carbon carrying an ``atomLabel`` (never a crash).

Deviation from upstream (documented): atom indices are remapped after "None"
predictions are dropped, so a dropped atom cannot corrupt the bonds of the
surviving ones (upstream keeps stale node indices there and would raise).
"""
import json
import os
from collections import Counter

from rdkit import Chem

__all__ = ["MolecularGraph", "load_abbreviations"]

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_BOND_MAP = {
    "SINGLE": Chem.rdchem.BondType.SINGLE,
    "DOUBLE": Chem.rdchem.BondType.DOUBLE,
    "TRIPLE": Chem.rdchem.BondType.TRIPLE,
    "AROMATIC": Chem.rdchem.BondType.AROMATIC,
    "DASHED": Chem.rdchem.BondType.SINGLE,
    "SOLID": Chem.rdchem.BondType.SINGLE,
}
_ELEMENTS = None


def load_abbreviations(path=None):
    """abbreviation text -> SMILES with an explicit ``*`` attachment atom."""
    path = path or os.path.join(_DATA_DIR, "abbreviations.json")
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for abbreviation, value in raw.items():
        smiles = value["smiles"]
        out[abbreviation] = smiles[0] if isinstance(smiles, list) else smiles
    return out


def _is_element_symbol(symbol):
    global _ELEMENTS
    if _ELEMENTS is None:
        _ELEMENTS = set()
        table = Chem.GetPeriodicTable()
        for z in range(1, 119):
            try:
                _ELEMENTS.add(table.GetElementSymbol(z))
            except Exception:
                pass
    return symbol in _ELEMENTS


class MolecularGraph:
    def __init__(self, config_dataset_graph, symbols_classes_atoms=None,
                 types_classes_bonds=None, abbreviations=None):
        if symbols_classes_atoms is None or types_classes_bonds is None:
            from .graph_constructor import load_vocabulary
            symbols_classes_atoms, types_classes_bonds = load_vocabulary(
                config_dataset_graph["nb_atoms_classes"],
                config_dataset_graph["nb_bonds_classes"])
        self.symbols_classes_atoms = symbols_classes_atoms
        self.types_classes_bonds = types_classes_bonds
        self.atoms_classes_symbols = {v: k for k, v in symbols_classes_atoms.items()}
        self.bonds_classes_types = {v: k for k, v in types_classes_bonds.items()}
        self.abbreviations = abbreviations if abbreviations is not None else load_abbreviations()
        self.atoms = []
        self.bonds = []
        self.abbreviations_used = []
        self.unresolved_abbreviations = []
        self.fallback = False
        self.n_dropped_atoms = 0
        self.n_dropped_bonds = 0

    # ------------------------------------------------------------------ #
    def from_predictions_nodes_only(self, atoms_classes, bonds_classes, structure,
                                    remove_none_predictions=True):
        """``MolecularGraph.from_predictions_nodes_only``."""
        nodes_types = structure.nodes_types.tolist()
        atom_node_indices = [i for i, t in enumerate(nodes_types) if t == 1]
        bond_node_indices = [i for i, t in enumerate(nodes_types) if t == 0]
        positions = structure.nodes_positions.tolist()

        removed_atom_indices = []
        self.atoms = []
        self.node_to_atom = {}
        for atom_index, atom_class in zip(atom_node_indices, atoms_classes):
            symbol = self.atoms_classes_symbols[atom_class]
            if remove_none_predictions and symbol in ("None", "Decoy"):
                removed_atom_indices.append(atom_index)
                continue
            self.node_to_atom[atom_index] = len(self.atoms)
            self.atoms.append({"index": atom_index, "class": atom_class,
                               "position": positions[atom_index], "type": 1})
        self.n_dropped_atoms = len(removed_atom_indices)

        neighboring_atoms_indices = structure.neighbors_of_bond_nodes()

        self.bonds = []
        dropped = 0
        for bond_index, bond_class in zip(bond_node_indices, bonds_classes):
            bond_type = self.bonds_classes_types[bond_class]
            if remove_none_predictions and bond_type in ("None", "Decoy"):
                dropped += 1
                continue
            neighbors = neighboring_atoms_indices.get(bond_index, [])
            if len(neighbors) != 2:
                dropped += 1
                continue
            if remove_none_predictions and (neighbors[0] in removed_atom_indices
                                            or neighbors[1] in removed_atom_indices):
                dropped += 1
                continue
            self.bonds.append({
                "index": [self.node_to_atom[neighbors[0]],
                          self.node_to_atom[neighbors[1]]],
                "class": bond_class,
                "position": positions[bond_index],
                "type": 0,
            })
        self.n_dropped_bonds = dropped
        return self

    # ------------------------------------------------------------------ #
    def _add_atoms(self, molecule):
        for atom in self.atoms:
            symbol = self.atoms_classes_symbols[atom["class"]]
            atom["_expand"] = False
            if "," in symbol:  # charged atom, e.g. "N,1"
                element, charge = symbol.split(",")
                rd_atom = Chem.Atom(element)
                rd_atom.SetFormalCharge(int(charge))
            elif symbol == "R":
                rd_atom = Chem.Atom("C")
                rd_atom.SetProp("atomLabel", "abb")
            elif _is_element_symbol(symbol):
                rd_atom = Chem.Atom(symbol)
            else:
                # abbreviation class (CF3 / OMe / Ph / ...): placeholder carbon,
                # expanded into its sub-molecule in ``_expand_abbreviations``.
                rd_atom = Chem.Atom("C")
                rd_atom.SetProp("atomLabel", symbol)
                atom["_expand"] = True
            molecule.AddAtom(rd_atom)
            atom["_rdkit_index"] = molecule.GetNumAtoms() - 1
        return molecule

    def _add_bonds(self, molecule):
        for bond in self.bonds:
            bond_type = self.bonds_classes_types[bond["class"]]
            molecule.AddBond(bond["index"][0], bond["index"][1], _BOND_MAP[bond_type])
        return molecule

    # ------------------------------------------------------------------ #
    def to_rdkit(self, _postprocessed_isolated_atoms=False):
        if not self.atoms:
            self.fallback = True
            return Chem.MolFromSmiles("C")

        molecule = Chem.RWMol()
        self._add_atoms(molecule)
        self._add_bonds(molecule)
        molecule = molecule.GetMol()

        molecule = self._expand_abbreviations(molecule)
        molecule = self.postprocess_aromatic_bonds(molecule)
        molecule = self.postprocess_aromatic_rings(molecule)

        if len(Chem.GetMolFrags(molecule)) == 1:
            sanitized = self.try_sanitize_molecule(molecule)
            if sanitized is not False:
                return sanitized

        if not _postprocessed_isolated_atoms and len(Chem.GetMolFrags(molecule)) > 1:
            if self.remove_isolated_atoms():
                return self.to_rdkit(_postprocessed_isolated_atoms=True)

        print("Predicted molecule has multiple fragments")
        self.fallback = True
        return Chem.MolFromSmiles("C")

    # ------------------------------------------------------------------ #
    def _expand_abbreviations(self, molecule):
        """Replace abbreviation placeholder carbons by their sub-molecule."""
        for atom in self.atoms:
            if not atom.get("_expand"):
                continue
            symbol = self.atoms_classes_symbols[atom["class"]]
            if symbol not in self.abbreviations:
                self.unresolved_abbreviations.append(symbol)
                try:
                    Chem.SetAtomAlias(molecule.GetAtomWithIdx(atom["_rdkit_index"]),
                                      f"[{symbol}]")
                except Exception:
                    pass
                continue
            fragment = Chem.MolFromSmiles(self.abbreviations[symbol])
            if fragment is None:
                self.unresolved_abbreviations.append(symbol)
                continue
            try:
                molecule = self._merge_fragment(molecule, atom["_rdkit_index"], fragment)
                self.abbreviations_used.append(symbol)
            except Exception:
                self.unresolved_abbreviations.append(symbol)
        return molecule

    @staticmethod
    def _merge_fragment(molecule, atom_index, fragment):
        stars = [a.GetIdx() for a in fragment.GetAtoms() if a.GetSymbol() == "*"]
        if len(stars) != 1:
            raise ValueError("only single attachment-point abbreviations supported")
        star = stars[0]
        if len(fragment.GetAtomWithIdx(star).GetBonds()) != 1:
            raise ValueError("ambiguous attachment point")
        star_bond = fragment.GetAtomWithIdx(star).GetBonds()[0]
        anchor = (star_bond.GetEndAtomIdx() if star_bond.GetBeginAtomIdx() == star
                  else star_bond.GetBeginAtomIdx())
        bond_order = star_bond.GetBondType()
        n_atoms_frag = fragment.GetNumAtoms()
        editable = Chem.RWMol(fragment)
        editable.RemoveAtom(star)
        fragment = editable.GetMol()
        anchor -= int(anchor > star)

        neighbors = [n.GetIdx() for n in molecule.GetAtomWithIdx(atom_index).GetNeighbors()]
        if len(neighbors) > 1:
            raise ValueError("placeholder atom needs exactly one neighbour")

        offset = molecule.GetNumAtoms()
        combined = Chem.RWMol(Chem.CombineMols(molecule, fragment))
        if neighbors:
            combined.AddBond(neighbors[0], offset + anchor, bond_order)
        # drop the placeholder atom (it is now redundant)
        combined.RemoveAtom(atom_index)
        return combined.GetMol()

    # ------------------------------------------------------------------ #
    @staticmethod
    def try_sanitize_molecule(molecule):
        try:
            Chem.SanitizeMol(molecule,
                             sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            return molecule
        except Exception as exc:
            print(f"The predicted molecule can not be sanitized: {exc}")
            return False

    @staticmethod
    def postprocess_aromatic_bonds(molecule):
        for bond in molecule.GetBonds():
            try:
                if str(bond.GetBondType()) == "AROMATIC" and not bond.IsInRing():
                    bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                    molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).SetIsAromatic(False)
                    molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).SetIsAromatic(False)
            except Exception as exc:
                print(f"Error in aromatic post-processing: {exc}")
        return molecule

    @staticmethod
    def postprocess_aromatic_rings(molecule):
        ring_info = molecule.GetRingInfo()
        for bonds_ring in ring_info.BondRings():
            types = [str(molecule.GetBondWithIdx(b).GetBondType()) for b in bonds_ring]
            if not types:
                continue
            most_common_type = Counter(types).most_common(1)[0][0]
            if most_common_type == "AROMATIC" and any(t != "AROMATIC" for t in types):
                for b in bonds_ring:
                    molecule.GetBondWithIdx(b).SetBondType(
                        Chem.rdchem.BondType.AROMATIC)
        return molecule

    def remove_isolated_atoms(self):
        involved = set()
        for bond in self.bonds:
            involved.update(bond["index"])
        if len(involved) == len(self.atoms):
            return False
        keep = sorted(involved)
        remap = {old: new for new, old in enumerate(keep)}
        self.atoms = [self.atoms[i] for i in keep]
        self.bonds = [{"index": [remap[b["index"][0]], remap[b["index"][1]]],
                       "class": b["class"], "position": b["position"], "type": 0}
                      for b in self.bonds
                      if b["index"][0] in remap and b["index"][1] in remap]
        return True

    # ------------------------------------------------------------------ #
    def to_smiles(self):
        molecule = self.to_rdkit()
        if molecule is None:
            return None, None
        try:
            smiles = Chem.MolToSmiles(molecule)
        except Exception as exc:
            print(f"MolToSmiles failed: {exc}")
            smiles = None
        molblock = None
        try:
            molblock = Chem.MolToMolBlock(molecule, kekulize=False)
        except Exception:
            pass
        return smiles, molblock
