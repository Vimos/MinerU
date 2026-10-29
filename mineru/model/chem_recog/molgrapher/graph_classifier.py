"""MolGrapher graph classifier: resnet50 backbone + 4x GCNConv + 2 heads.

Port of ``molgrapher/models/graph_classifier.py`` (``GraphClassifier``),
``torch_geometric`` replaced by :class:`ocsr.molgrapher.gcn.GCNConv` and PyG
``Data``/``Batch`` replaced by :class:`GraphStructure`.

Checkpoint ``models/graph_classifier/gc_gcn_model.ckpt`` loads with zero
missing / zero unexpected keys (339 tensors).  ``gc_gcn_model`` uses
141 atom classes / 5 bond classes and ``gcn_on=True``; it does **not** rely on
the PaddleOCR abbreviation detector (the 141-class vocabulary already labels
abbreviation groups such as ``CF3``/``OMe`` as single atoms).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GCNConv
from .resnet import resnet50

__all__ = ["Backbone", "GNN", "GraphClassifier", "get_point_of_trisection_coordinates"]


def get_point_of_trisection_coordinates(segment, ratio_b, ratio_e):
    x = (ratio_e * segment[0][0] + ratio_b * segment[1][0]) / (ratio_b + ratio_e)
    y = (ratio_e * segment[0][1] + ratio_b * segment[1][1]) / (ratio_b + ratio_e)
    return torch.tensor([x, y], dtype=torch.float32)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(output_layers=["layer4"], dilation_factor=2,
                                 conv1_stride=2)
        self.nb_filters = 2048
        self.features_maps_spatial_size = [64, 64]

    def forward(self, x):
        # (N, 3, 1024, 1024) -> (N, 2048, 256, 256)
        return self.backbone(x)


class GNN(nn.Module):
    def __init__(self, node_embedding_dimension, nb_atoms_classes, nb_bonds_classes,
                 gcn_on=True):
        super().__init__()
        self.node_embedding_dimension = node_embedding_dimension
        self.nb_filters_out = node_embedding_dimension
        self.gcn_on = gcn_on

        if gcn_on:
            self.nb_filters_out = node_embedding_dimension // 8
            self.conv1 = GCNConv(node_embedding_dimension, node_embedding_dimension // 4)
            self.conv2 = GCNConv(node_embedding_dimension // 4,
                                 node_embedding_dimension // 8)
            self.conv3 = GCNConv(node_embedding_dimension // 8,
                                 node_embedding_dimension // 8)
            self.conv4 = GCNConv(node_embedding_dimension // 8, self.nb_filters_out)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.2)

        self.mlp_atoms = nn.Sequential(
            nn.Linear(self.nb_filters_out, 256), nn.ReLU(), nn.Dropout(p=0.15),
            nn.Linear(256, nb_atoms_classes))
        self.mlp_bonds = nn.Sequential(
            nn.Linear(self.nb_filters_out, 256), nn.ReLU(), nn.Dropout(p=0.15),
            nn.Linear(256, nb_bonds_classes))

    def forward(self, x, edge_index, nodes_types):
        if self.gcn_on:
            for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
                x = conv(x, edge_index)
                x = self.dropout(self.relu(x))
        atoms_indices = (nodes_types == 1).nonzero(as_tuple=False).view(-1)
        bonds_indices = (nodes_types == 0).nonzero(as_tuple=False).view(-1)
        return [self.mlp_atoms(x[atoms_indices]), self.mlp_bonds(x[bonds_indices])]


class GraphClassifier(nn.Module):
    def __init__(self, nb_atoms_classes=141, nb_bonds_classes=5, gcn_on=True,
                 bond_nb_sampled_points=3):
        super().__init__()
        self.nb_atoms_classes = nb_atoms_classes
        self.nb_bonds_classes = nb_bonds_classes
        self.backbone = Backbone()
        self.node_embedding_dimension = self.backbone.nb_filters
        self.gnn = GNN(self.node_embedding_dimension, nb_atoms_classes,
                       nb_bonds_classes, gcn_on)
        self.bond_nb_sampled_points = bond_nb_sampled_points
        self.bond_fusion = False
        self.positional_embedding = False
        self.type_embedding = True
        self.atom_type_embedding = nn.Parameter(torch.randn(self.node_embedding_dimension))
        self.bond_type_embedding = nn.Parameter(torch.randn(self.node_embedding_dimension))

        # ``criterion_bonds`` carries a persistent (nb_bonds,) buffer in the
        # gc_gcn_model checkpoint (CrossEntropyLoss(weight=...)), so it is kept
        # to make the state dict match exactly.  The non-GCN variants were
        # trained with an unweighted criterion (upstream if/else on gcn_on).
        self.criterion_atoms = nn.CrossEntropyLoss()
        if gcn_on:
            self.criterion_bonds = nn.CrossEntropyLoss(
                weight=torch.tensor([1.0] * nb_bonds_classes))
        else:
            self.criterion_bonds = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------ #
    def _sample_node_embeddings(self, feature_maps, graph, image_size):
        """Visual embedding of every node (atoms + bond mid-point samples)."""
        map_node_sample_index = {}
        sample_index = 0
        for node_index, node_type in enumerate(graph.nodes_types.tolist()):
            if node_type == 1:
                map_node_sample_index[node_index] = sample_index
                sample_index += 1
            if node_type == 0:
                map_node_sample_index[node_index] = list(
                    range(sample_index, sample_index + self.bond_nb_sampled_points))
                sample_index += self.bond_nb_sampled_points
        nb_samples = sample_index

        neighboring_atoms_indices = graph.neighbors_of_bond_nodes()
        grid = torch.zeros(1, nb_samples, 1, 2, dtype=torch.float32,
                           device=feature_maps.device)
        H = image_size[2]
        W = image_size[1]

        for node_index, (x, y) in enumerate(graph.nodes_positions.tolist()):
            if int(graph.nodes_types[node_index]) == 1:
                grid[0, map_node_sample_index[node_index], 0, 0] = 2 * (x / (W - 1)) - 1
                grid[0, map_node_sample_index[node_index], 0, 1] = -(
                    2 * (((H - 1) - y) / (H - 1)) - 1)
            if int(graph.nodes_types[node_index]) == 0:
                bond_positions = [
                    graph.nodes_positions[neighboring_atoms_indices[node_index][0]].tolist(),
                    graph.nodes_positions[neighboring_atoms_indices[node_index][1]].tolist(),
                ]
                bond_sampling_positions = [
                    get_point_of_trisection_coordinates(bond_positions, ratio_b, ratio_e)
                    for ratio_b, ratio_e in zip(
                        range(1, self.bond_nb_sampled_points + 1),
                        range(self.bond_nb_sampled_points, 0, -1))]
                for (x_bond_sample, y_bond_sample), sample_idx in zip(
                        bond_sampling_positions, map_node_sample_index[node_index]):
                    grid[0, sample_idx, 0, 0] = 2 * (x_bond_sample / (W - 1)) - 1
                    grid[0, sample_idx, 0, 1] = -(
                        2 * (((H - 1) - y_bond_sample) / (H - 1)) - 1)

        sampled = F.grid_sample(feature_maps, grid, mode="bilinear",
                                align_corners=False)[0, :, :, 0].permute(1, 0)

        # Merge bond sampled features (mean over the 3 sample points)
        bonds_image_features = []
        for node_index in range(graph.num_nodes):
            if int(graph.nodes_types[node_index]) == 0:
                bonds_image_features.append(sampled[map_node_sample_index[node_index]])
        if len(bonds_image_features) > 0:
            bonds_image_features = torch.stack(bonds_image_features).mean(1)

        node_embeddings = []
        bond_index = 0
        for node_index in range(graph.num_nodes):
            if int(graph.nodes_types[node_index]) == 1:
                emb = sampled[map_node_sample_index[node_index]] + self.atom_type_embedding
            else:
                emb = bonds_image_features[bond_index] + self.bond_type_embedding
                bond_index += 1
            node_embeddings.append(emb)
        return torch.stack(node_embeddings)

    def forward(self, images, graphs):
        """images: (B,3,S,S); graphs: list[GraphStructure] -> list[[atoms, bonds]]."""
        feature_maps = self.backbone(images)["layer4"]
        image_size = [images.shape[1], images.shape[3], images.shape[2]]
        predictions = []
        for data_index, graph in enumerate(graphs):
            x = self._sample_node_embeddings(
                feature_maps[data_index:data_index + 1], graph, image_size)
            predictions.append(
                self.gnn(x, graph.edge_index.to(x.device), graph.nodes_types.to(x.device)))
        return predictions

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, images, graphs):
        """-> (per-graph top-1 atom/bond classes, per-graph confidence)."""
        outs = self.forward(images, graphs)
        all_atoms, all_bonds, confidences = [], [], []
        for graph, (atoms_pred, bonds_pred) in zip(graphs, outs):
            atoms_pred = atoms_pred.detach().cpu()
            bonds_pred = bonds_pred.detach().cpu()
            atoms_cls = atoms_pred.argmax(-1).tolist()
            bonds_cls = bonds_pred.argmax(-1).tolist()
            conf = 1.0
            if len(atoms_cls):
                conf = min(conf, float(atoms_pred.softmax(-1).max(-1).values.min()))
            if len(bonds_cls):
                conf = min(conf, float(bonds_pred.softmax(-1).max(-1).values.min()))
            all_atoms.append(atoms_cls)
            all_bonds.append(bonds_cls)
            confidences.append(round(conf, 3))
        return all_atoms, all_bonds, confidences
