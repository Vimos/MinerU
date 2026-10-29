"""Character-level tokenizer with (x, y) coordinate bins, MolScribe-compatible.

Only the *decoding* half of MolScribe's ``CharTokenizer`` is needed for
inference, so the (SmilesPE-dependent) encoding half is dropped.

Sequence grammar (``sep_xy=True``, ``continuous_coords=False``)::

    <sos>  [x_id y_id char ...]  <eos>

where ``x_id = offset + round(x*(maxx-1))``, ``y_id = offset + maxx +
round(y*(maxy-1))`` and ``char`` indexes the character vocabulary.
"""
import json

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"
MASK = "<mask>"
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 4


class CharTokenizer:
    def __init__(self, input_size=100, path=None, sep_xy=False):
        self.stoi = {}
        self.itos = {}
        self.maxx = input_size
        self.maxy = input_size
        self.sep_xy = sep_xy
        self.special_tokens = [PAD, SOS, EOS, UNK, MASK]
        if path:
            with open(path) as f:
                self.stoi = json.load(f)
            self.itos = {v: k for k, v in self.stoi.items()}

    # ---- vocab geometry ---------------------------------------------------
    def __len__(self):
        if self.sep_xy:
            return self.offset + self.maxx + self.maxy
        return self.offset + max(self.maxx, self.maxy)

    @property
    def offset(self):
        return len(self.stoi)

    @property
    def output_constraint(self):
        return True

    def is_x(self, x):
        return self.offset <= x < self.offset + self.maxx

    def is_y(self, y):
        if self.sep_xy:
            return self.offset + self.maxx <= y < self.offset + self.maxx + self.maxy
        return self.offset <= y < self.offset + self.maxy

    def is_symbol(self, s):
        return len(self.special_tokens) <= s < self.offset or s == UNK_ID

    def is_atom_token(self, token):
        return token.isalpha() or token.startswith("[") or token == "*" or token == UNK

    def is_atom(self, id):
        return self.is_symbol(id) and self.is_atom_token(self.itos[id])

    def id_to_x(self, id):
        return (id - self.offset) / (self.maxx - 1)

    def id_to_y(self, id):
        if self.sep_xy:
            return (id - self.offset - self.maxx) / (self.maxy - 1)
        return (id - self.offset) / (self.maxy - 1)

    # ---- decoding ---------------------------------------------------------
    def predict_caption(self, sequence):
        caption = ""
        for i in sequence:
            if i == EOS_ID or i == PAD_ID:
                break
            caption += self.itos[i]
        return caption

    def get_output_mask(self, id):
        """Constraint used during greedy decoding: after an x-bin only a y-bin is
        legal, after a y-bin only a symbol is legal."""
        mask = [False] * len(self)
        if self.is_x(id):
            return [True] * (self.offset + self.maxx) + [False] * self.maxy
        if self.is_y(id):
            return [False] * self.offset + [True] * (self.maxx + self.maxy)
        return mask

    def sequence_to_smiles(self, sequence):
        """Same output contract as MolScribe's ``CharTokenizer.sequence_to_smiles``."""
        smiles = ""
        coords, symbols, indices = [], [], []
        i = 0
        while i < len(sequence):
            label = sequence[i]
            if label == EOS_ID or label == PAD_ID:
                break
            if self.is_x(label) or self.is_y(label):
                i += 1
                continue
            if not self.is_atom(label):
                smiles += self.itos[label]
                i += 1
                continue
            if self.itos[label] == "[":
                j = i + 1
                while j < len(sequence):
                    if not self.is_symbol(sequence[j]):
                        break
                    if self.itos[sequence[j]] == "]":
                        j += 1
                        break
                    j += 1
            else:
                if (i + 1 < len(sequence) and self.itos[label] == "C"
                        and self.is_symbol(sequence[i + 1]) and self.itos[sequence[i + 1]] == "l") or (
                        i + 1 < len(sequence) and self.itos[label] == "B"
                        and self.is_symbol(sequence[i + 1]) and self.itos[sequence[i + 1]] == "r"):
                    j = i + 2
                else:
                    j = i + 1
            token = "".join(self.itos[sequence[k]] for k in range(i, j))
            smiles += token
            if j + 2 < len(sequence) and self.is_x(sequence[j]) and self.is_y(sequence[j + 1]):
                x = self.id_to_x(sequence[j])
                y = self.id_to_y(sequence[j + 1])
                coords.append([x, y])
                symbols.append(token)
                indices.append(j + 2)
                i = j + 2
            else:
                i = j
        return {"smiles": smiles, "symbols": symbols, "indices": indices, "coords": coords}


def get_tokenizer(vocab_file, coord_bins=64, sep_xy=True):
    return {"chartok_coords": CharTokenizer(input_size=coord_bins, path=vocab_file, sep_xy=sep_xy)}
