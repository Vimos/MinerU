"""MolScribe model: Swin encoder + autoregressive Transformer decoder.

Clean-room re-implementation with 1:1 parameter names against the official
checkpoint ``swin_base_char_aux_1m680k.pth``::

    encoder.transformer.*                    (Swin-B, patch4 / window12 / 384)
    decoder.chartok_coords.enc_trans_layer.0.*
    decoder.chartok_coords.embeddings.*
    decoder.chartok_coords.decoder.*
    decoder.chartok_coords.output_layer.*
    decoder.edges.mlp.*                      (bond-type graph head)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin import SwinTransformer
from .decoder import TransformerDecoder, Embeddings
from .tokenizer import SOS_ID, EOS_ID, PAD_ID, CharTokenizer


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    """Swin-B base encoder, 384x384 input, patch 4, window 12."""

    def __init__(self, img_size=384, embed_dim=128, depths=(2, 2, 18, 2),
                 num_heads=(4, 8, 16, 32), window_size=12, mlp_ratio=4.0,
                 drop_path_rate=0.1, use_checkpoint=False):
        super().__init__()
        self.model_type = "swin"
        self.transformer = SwinTransformer(
            img_size=img_size, patch_size=4, in_chans=3, num_classes=0,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=True,
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            use_checkpoint=use_checkpoint,
        )
        self.n_features = self.transformer.num_features  # 1024
        self.transformer.head = nn.Identity()

    def forward(self, x):
        """x: (B, 3, 384, 384) -> features (B, 144, 1024), hiddens list."""
        features, hiddens = self.transformer(x)
        return features, hiddens


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #
class TransformerDecoderAR(nn.Module):
    """Autoregressive token decoder (``chartok_coords`` head)."""

    def __init__(self, encoder_dim=1024, dec_hidden_size=256, dec_num_layers=6,
                 dec_attn_heads=8, vocab_size=229, dropout=0.1,
                 attention_dropout=0.1, coord_bins=64):
        super().__init__()
        self.enc_trans_layer = nn.Sequential(nn.Linear(encoder_dim, dec_hidden_size))
        self.embeddings = Embeddings(
            word_vec_size=dec_hidden_size,
            word_vocab_size=vocab_size,
            word_padding_idx=PAD_ID,
            position_encoding=True,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=dec_num_layers,
            d_model=dec_hidden_size,
            heads=dec_attn_heads,
            d_ff=dec_hidden_size * 4,
            dropout=dropout,
            attention_dropout=attention_dropout,
            pos_ffn_activation_fn="gelu",
        )
        self.output_layer = nn.Linear(dec_hidden_size, vocab_size, bias=True)

    def enc_transform(self, encoder_out):
        b, _, dim = encoder_out.size()
        return self.enc_trans_layer(encoder_out.view(b, -1, dim))

    def dec_embedding(self, tgt, step=None):
        pad_idx = self.embeddings.word_padding_idx
        tgt_pad_mask = tgt.data.eq(pad_idx).transpose(1, 2)  # (B, 1, T)
        return self.embeddings(tgt, step=step), tgt_pad_mask

    def forward(self, memory_bank, tgt):
        """memory_bank: (B, S, D) already projected; tgt: (B, T, 1)."""
        tgt_emb, tgt_pad_mask = self.dec_embedding(tgt)
        dec_out, *_ = self.decoder(tgt_emb, memory_bank, tgt_pad_mask=tgt_pad_mask)
        return self.output_layer(dec_out), dec_out


class GraphPredictor(nn.Module):
    """Bond-type classifier over every ordered atom pair (7 classes)."""

    def __init__(self, decoder_dim, coords=False):
        super().__init__()
        self.coords = coords
        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, 7),
        )
        if coords:
            self.coords_mlp = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim), nn.GELU(),
                nn.Linear(decoder_dim, 2),
            )

    def forward(self, hidden, indices=None):
        b, l, dim = hidden.size()
        if indices is None:
            hidden = hidden[:, [i for i in range(3, l, 3)]]
        else:
            batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
            hidden = hidden[batch_id, indices.view(-1)].view(b, -1, dim)
        b, l, dim = hidden.size()
        results = {}
        hh = torch.cat([hidden.unsqueeze(2).expand(b, l, l, dim),
                        hidden.unsqueeze(1).expand(b, l, l, dim)], dim=3)
        results["edges"] = self.mlp(hh).permute(0, 3, 1, 2)
        if self.coords:
            results["coords"] = self.coords_mlp(hidden)
        return results


class Decoder(nn.Module):
    """Wrapper holding one AR decoder per text format plus the edge head."""

    def __init__(self, formats=("chartok_coords", "edges"), encoder_dim=1024,
                 dec_hidden_size=256, dec_num_layers=6, dec_attn_heads=8,
                 vocab_size=229, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.formats = list(formats)
        modules = {}
        for fmt in self.formats:
            if fmt == "edges":
                modules["edges"] = GraphPredictor(dec_hidden_size, coords=False)
            else:
                modules[fmt] = TransformerDecoderAR(
                    encoder_dim=encoder_dim, dec_hidden_size=dec_hidden_size,
                    dec_num_layers=dec_num_layers, dec_attn_heads=dec_attn_heads,
                    vocab_size=vocab_size, dropout=dropout,
                    attention_dropout=attention_dropout,
                )
        self.decoder = nn.ModuleDict(modules)


# --------------------------------------------------------------------------- #
# Full model + greedy inference
# --------------------------------------------------------------------------- #
def get_edge_prediction(edge_prob):
    """Symmetrise the (k, k, 7) bond probability tensor and arg-max it."""
    n = len(edge_prob)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(5):
                edge_prob[i][j][k] = (edge_prob[i][j][k] + edge_prob[j][i][k]) / 2
                edge_prob[j][i][k] = edge_prob[i][j][k]
            edge_prob[i][j][5] = (edge_prob[i][j][5] + edge_prob[j][i][6]) / 2
            edge_prob[i][j][6] = (edge_prob[i][j][6] + edge_prob[j][i][5]) / 2
            edge_prob[j][i][5] = edge_prob[i][j][6]
            edge_prob[j][i][6] = edge_prob[i][j][5]
    prediction = np.argmax(edge_prob, axis=2).tolist()
    score = np.max(edge_prob, axis=2).tolist()
    return prediction, score


class MolScribeNet(nn.Module):
    def __init__(self, img_size=384, encoder_dim=1024, embed_dim=128,
                 depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), window_size=12,
                 dec_hidden_size=256, dec_num_layers=6, dec_attn_heads=8,
                 vocab_size=229, dropout=0.1, attention_dropout=0.1,
                 formats=("chartok_coords", "edges")):
        super().__init__()
        self.encoder = Encoder(img_size=img_size, embed_dim=embed_dim,
                               depths=depths, num_heads=num_heads,
                               window_size=window_size)
        self.decoder = Decoder(formats=formats, encoder_dim=encoder_dim,
                               dec_hidden_size=dec_hidden_size,
                               dec_num_layers=dec_num_layers,
                               dec_attn_heads=dec_attn_heads,
                               vocab_size=vocab_size, dropout=dropout,
                               attention_dropout=attention_dropout)

    @torch.no_grad()
    def decode(self, images, tokenizer: CharTokenizer, max_length=480,
               min_length=1):
        """Greedy decode a batch of preprocessed images.

        Returns a list of dicts with keys ``smiles`` (raw token string),
        ``symbols``, ``coords``, ``indices``, ``edges``, ``token_scores``.
        """
        features, _ = self.encoder(images)
        ar = self.decoder.decoder["chartok_coords"]
        memory_bank = ar.enc_transform(features)  # (B, 144, 256)
        batch_size = memory_bank.size(0)
        device = memory_bank.device

        # --- token loop (full-prefix recompute: exact same values as MolScribe's
        # stepwise decoding with KV cache, but far simpler) -------------------
        seqs = [[SOS_ID] for _ in range(batch_size)]
        scores = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for step in range(max_length):
            tgt = torch.tensor([s for s in seqs], dtype=torch.long,
                               device=device).unsqueeze(-1)  # (B, T, 1)
            tgt_emb, tgt_pad_mask = ar.dec_embedding(tgt)
            dec_out, *_ = ar.decoder(tgt_emb, memory_bank, tgt_pad_mask=tgt_pad_mask)
            logits = ar.output_layer(dec_out[:, -1])
            log_probs = F.log_softmax(logits, dim=-1)

            if tokenizer.output_constraint:
                out_mask = torch.tensor(
                    [tokenizer.get_output_mask(s[-1]) for s in seqs],
                    dtype=torch.bool, device=device)
                log_probs = log_probs.masked_fill(out_mask, -10000)
            if step < min_length:  # identical to DecodeStrategy.ensure_min_length
                log_probs[:, EOS_ID] = -1e20

            best, nxt = log_probs.topk(1, dim=-1)  # greedy
            for b in range(batch_size):
                seqs[b].append(int(nxt[b, 0]))
                scores[b].append(float(best[b, 0]))
                if int(nxt[b, 0]) == EOS_ID:
                    finished[b] = True
            if bool(finished.all()):
                break

        # --- re-run the (causal) decoder over the finished sequence to get the
        # per-token hidden states used by the edge head ----------------------
        results = []
        for b in range(batch_size):
            sequence = seqs[b][1:]  # drop <sos>, keep <eos> like MolScribe
            parsed = tokenizer.sequence_to_smiles(sequence)

            inputs = [SOS_ID] + sequence[:-1] if len(sequence) > 1 else [SOS_ID]
            tgt = torch.tensor(inputs, dtype=torch.long, device=device).view(1, -1, 1)
            tgt_emb, tgt_pad_mask = ar.dec_embedding(tgt)
            dec_out, *_ = ar.decoder(tgt_emb, memory_bank[b:b + 1],
                                     tgt_pad_mask=tgt_pad_mask)
            hidden = dec_out  # (1, len(inputs), 256)

            edges = []
            if len(parsed["symbols"]) > 0:
                idx = torch.tensor(parsed["indices"], dtype=torch.long,
                                   device=device).view(1, -1)
                pred = self.decoder.decoder["edges"](hidden, idx)
                prob = F.softmax(pred["edges"].squeeze(0).permute(1, 2, 0), dim=2)
                edges, _ = get_edge_prediction(prob.tolist())

            results.append({
                "smiles": parsed["smiles"],
                "symbols": parsed["symbols"],
                "coords": parsed["coords"],
                "indices": parsed["indices"],
                "edges": edges,
                "sequence": sequence,
                "token_scores": scores[b],
            })
        return results
