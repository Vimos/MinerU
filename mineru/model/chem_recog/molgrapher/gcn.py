"""``torch_geometric.nn.GCNConv`` re-implemented in plain torch.

Only the configuration used by MolGrapher is supported, and it is the
*torch_geometric 2.x default* one, which is what the released checkpoints were
trained with:

    GCNConv(in, out)  ==  normalize=True, add_self_loops=True, improved=False,
                          cached=False, bias=True

i.e. ``y = D^-1/2 (A + I) D^-1/2 X W + b`` with a mean-free "add" aggregation
(no degree averaging), ``W`` = ``lin.weight`` and ``b`` = ``bias``.  The
parameter names match PyG exactly (``lin.weight`` / ``bias``), so checkpoints
load unchanged.
"""
import torch
import torch.nn as nn

__all__ = ["GCNConv", "gcn_norm"]


def gcn_norm(edge_index, num_nodes, dtype=None, device=None):
    """``torch_geometric.nn.conv.gcn_conv.gcn_norm`` (add_self_loops=True)."""
    dtype = dtype or torch.float32
    device = device or edge_index.device

    loop_index = torch.arange(num_nodes, dtype=torch.long, device=device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)

    edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=device)
    deg = torch.zeros(num_nodes, dtype=dtype, device=device)
    deg = deg.index_add_(0, edge_index[0], edge_weight)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    edge_weight = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
    return edge_index, edge_weight


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, edge_index):
        edge_index, edge_weight = gcn_norm(edge_index, x.size(0), x.dtype, x.device)
        out = self.lin(x)
        msg = out[edge_index[1]] * edge_weight.unsqueeze(-1)
        agg = torch.zeros_like(out)
        agg = agg.index_add_(0, edge_index[0], msg)
        if self.bias is not None:
            agg = agg + self.bias
        return agg
