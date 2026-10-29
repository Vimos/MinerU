"""Shared chemistry helpers (rdkit only)."""
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

__all__ = ["mol_from_smiles", "canonical_smiles", "tanimoto", "is_valid",
           "exact_match", "formula"]


def mol_from_smiles(smiles, sanitize=True):
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        return Chem.MolFromSmiles(smiles.strip(), sanitize=sanitize)
    except Exception:
        return None


def canonical_smiles(smiles):
    """Canonical (isomeric) SMILES, or ``None`` when unparsable."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def is_valid(smiles):
    return mol_from_smiles(smiles) is not None


_MORGAN = None


def tanimoto(smiles_a, smiles_b, radius=2, n_bits=2048):
    """Morgan (ECFP4) Tanimoto similarity of two SMILES; 0.0 if unparsable."""
    global _MORGAN
    mol_a, mol_b = mol_from_smiles(smiles_a), mol_from_smiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    if _MORGAN is None or _MORGAN[0] != (radius, n_bits):
        _MORGAN = ((radius, n_bits),
                   rdFingerprintGenerator.GetMorganGenerator(radius=radius,
                                                             fpSize=n_bits))
    gen = _MORGAN[1]
    from rdkit import DataStructs

    return float(DataStructs.TanimotoSimilarity(gen.GetFingerprint(mol_a),
                                                gen.GetFingerprint(mol_b)))


def exact_match(smiles_a, smiles_b):
    ca, cb = canonical_smiles(smiles_a), canonical_smiles(smiles_b)
    return ca is not None and ca == cb


def formula(smiles):
    mol = mol_from_smiles(smiles)
    return None if mol is None else Chem.rdMolDescriptors.CalcMolFormula(mol)
