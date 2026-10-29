"""Pure-PyTorch re-implementation of the OpenNMT ``TransformerDecoder`` used by
MolScribe, with *identical* parameter names so the official checkpoints map 1:1.

Mapped names (per layer ``i``)::

    transformer_layers.{i}.self_attn.{linear_keys,linear_values,linear_query,final_linear}.{weight,bias}
    transformer_layers.{i}.context_attn.{linear_keys,linear_values,linear_query,final_linear}.{weight,bias}
    transformer_layers.{i}.feed_forward.{w_1,w_2}.{weight,bias}
    transformer_layers.{i}.feed_forward.layer_norm.{weight,bias}
    transformer_layers.{i}.layer_norm_1.{weight,bias}
    transformer_layers.{i}.layer_norm_2.{weight,bias}
    layer_norm.{weight,bias}

and for the word embeddings::

    embeddings.make_embedding.emb_luts.0.weight
    embeddings.make_embedding.pe.pe            (non-trainable buffer)

No ``onmt`` import; ~200 lines instead of a 5k-line dependency.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Attention / feed-forward blocks (OpenNMT semantics)
# --------------------------------------------------------------------------- #
class MultiHeadedAttention(nn.Module):
    """OpenNMT ``MultiHeadedAttention`` (scaled dot product, no relative pos)."""

    def __init__(self, head_count, model_dim, dropout=0.1):
        super().__init__()
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(head_count * self.dim_per_head, model_dim)

    def _shape(self, x):
        b = x.size(0)
        return x.view(b, -1, self.head_count, self.dim_per_head).transpose(1, 2)

    def _unshape(self, x):
        b = x.size(0)
        return x.transpose(1, 2).contiguous().view(b, -1, self.head_count * self.dim_per_head)

    def forward(self, key, value, query, mask=None):
        """key/value/query: (B, T, D); mask: bool (B, T_q, T_k) where True = mask."""
        key = self._shape(self.linear_keys(key))
        value = self._shape(self.linear_values(value))
        query = self._shape(self.linear_query(query)) / math.sqrt(self.dim_per_head)

        scores = torch.matmul(query, key.transpose(2, 3))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).expand_as(scores), -1e18)
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        context = self._unshape(torch.matmul(attn, value))
        return self.final_linear(context), attn


class PositionwiseFeedForward(nn.Module):
    """OpenNMT ``PositionwiseFeedForward``.

    Note the *internal residual*: ``w_2(act(w_1(LN(x)))) + x``.  OpenNMT's FFN
    adds its input back, and MolScribe's decoder layer passes ``drop(mid) +
    query`` in, so the layer output is ``w_2(...) + mid + query``.
    """

    def __init__(self, model_dim, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.w_1 = nn.Linear(model_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(activation)

    def forward(self, x):
        inter = self.dropout_1(self.activation(self.w_1(self.layer_norm(x))))
        return self.dropout_2(self.w_2(inter)) + x


# --------------------------------------------------------------------------- #
# Decoder layer (pre-norm, as in OpenNMT's TransformerDecoderLayer)
# --------------------------------------------------------------------------- #
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 pos_ffn_activation_fn="gelu"):
        super().__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=attention_dropout)
        self.context_attn = MultiHeadedAttention(heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, pos_ffn_activation_fn)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, memory_bank, src_pad_mask=None, tgt_pad_mask=None,
                future=False):
        """inputs: (B, T, D) target embeddings, memory_bank: (B, S, D) encoder states."""
        dec_mask = None
        if inputs.size(1) > 1:
            # (B, T, T) causal (+ padding) mask; True means "masked out"
            tgt_len = tgt_pad_mask.size(-1)
            if not future:
                future_mask = torch.triu(
                    torch.ones(tgt_len, tgt_len, device=inputs.device, dtype=torch.bool), diagonal=1
                ).view(1, tgt_len, tgt_len)
                dec_mask = tgt_pad_mask.bool() | future_mask
            else:
                dec_mask = tgt_pad_mask.bool()

        inputs_norm = self.layer_norm_1(inputs)
        query, _ = self.self_attn(inputs_norm, inputs_norm, inputs_norm, mask=dec_mask)
        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attns = self.context_attn(memory_bank, memory_bank, query_norm, mask=src_pad_mask)
        output = self.feed_forward(self.drop(mid) + query)
        return output, attns


# --------------------------------------------------------------------------- #
# Embeddings with sinusoidal positional encoding (OpenNMT flavour)
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal PE.  ``pe`` is a persistent buffer of shape (max_len, 1, dim),
    exactly as in the checkpoint (``...pe.pe``, shape (5000, 1, 256))."""

    def __init__(self, dropout, dim, max_len=5000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dim")
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, dim)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """emb: (B, T, dim).  Positions are ``step .. step+T-1``."""
        step = step or 0
        emb = emb * math.sqrt(self.dim)
        pe = self.pe[step:step + emb.size(1)].transpose(0, 1)  # (1, T, dim)
        return self.dropout(emb + pe)


class Embeddings(nn.Module):
    """Word embeddings (+PE).  Keeps OpenNMT's module names for weight mapping."""

    def __init__(self, word_vec_size, word_vocab_size, word_padding_idx,
                 position_encoding=True, dropout=0.0):
        super().__init__()
        self.word_padding_idx = word_padding_idx
        self.word_vec_size = word_vec_size
        self.embedding_size = word_vec_size
        self.position_encoding = position_encoding
        self.make_embedding = nn.Sequential()
        # named `emb_luts` (ModuleList) so the key is emb_luts.0.weight
        self.make_embedding.add_module(
            "emb_luts", nn.ModuleList([nn.Embedding(word_vocab_size, word_vec_size,
                                                    padding_idx=word_padding_idx)])
        )
        if position_encoding:
            self.make_embedding.add_module("pe", PositionalEncoding(dropout, word_vec_size))

    @property
    def word_lut(self):
        return self.make_embedding[0][0]

    def forward(self, source, step=None):
        """source: (B, T, 1) long tensor -> (B, T, dim)."""
        x = self.word_lut(source[:, :, 0])
        if self.position_encoding:
            x = self.make_embedding[1](x, step=step)
        return x


# --------------------------------------------------------------------------- #
# Full decoder stack
# --------------------------------------------------------------------------- #
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout=0.1,
                 attention_dropout=0.1, pos_ffn_activation_fn="gelu"):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.transformer_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, heads, d_ff, dropout, attention_dropout,
                                    pos_ffn_activation_fn)
            for _ in range(num_layers)
        ])

    def forward(self, tgt_emb, memory_bank, src_pad_mask=None, tgt_pad_mask=None):
        """Returns (output (B,T,D), attns dict, per-layer hidden list)."""
        batch_size, src_len, _ = memory_bank.size()
        device = memory_bank.device
        if src_pad_mask is None:
            src_pad_mask = torch.zeros(batch_size, 1, src_len, dtype=torch.bool, device=device)
        tgt_len = tgt_emb.size(1)
        if tgt_pad_mask is None:
            tgt_pad_mask = torch.zeros(batch_size, 1, tgt_len, dtype=torch.bool, device=device)

        output = tgt_emb
        hiddens = []
        attn = None
        for layer in self.transformer_layers:
            output, attn = layer(output, memory_bank, src_pad_mask, tgt_pad_mask)
            hiddens.append(output)
        output = self.layer_norm(output)
        return output, {"std": attn}, hiddens
