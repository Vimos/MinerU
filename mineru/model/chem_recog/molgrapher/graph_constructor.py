"""Dense-graph construction from heatmap keypoints (no cv2, no torch_geometric).

Port of ``molgrapher/models/graph_constructor.py`` + the nodes-only part of
``molgrapher/utils/utils_graph.py`` (``MolecularGraph``).  The PyG ``Data``
object is replaced by :class:`GraphStructure`, a tiny container with the same
fields (``edge_index`` is still a ``(2, E)`` long tensor with both edge
directions, as PyG's ``DuplicateEdges`` produced).
"""
import json
import math
import os
from math import atan

import numpy as np
import torch

from .geometry import box_points, crop_pad, point_in_polygon

__all__ = ["GraphStructure", "GraphConstructor"]

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_vocabulary(nb_atoms_classes=141, nb_bonds_classes=5, data_dir=None):
    data_dir = data_dir or _DATA_DIR
    with open(os.path.join(data_dir, f"vocabulary_atoms_{nb_atoms_classes}.json")) as f:
        atoms = json.load(f)
    with open(os.path.join(data_dir, f"vocabulary_bonds_{nb_bonds_classes}.json")) as f:
        bonds = json.load(f)
    return atoms, bonds


class GraphStructure:
    """Minimal stand-in for ``torch_geometric.data.Data`` (nodes-only graph)."""

    def __init__(self, edge_index, nodes_positions, nodes_classes, nodes_types):
        self.edge_index = edge_index
        self.nodes_positions = nodes_positions
        self.nodes_classes = nodes_classes
        self.nodes_types = nodes_types
        self.x = None
        self.num_nodes = int(nodes_types.numel())
        self.num_graphs = 1

    # PyG-like helpers used by the classifier / graph builder
    def neighbors_of_bond_nodes(self):
        """bond-node index -> [atom-node index, ...] (upstream defaultdict logic)."""
        nb = {}
        for index_b, index_e in self.edge_index.t().tolist():
            if int(self.nodes_types[index_b]) == 0:
                if index_e not in nb.setdefault(index_b, []):
                    nb[index_b].append(index_e)
            if int(self.nodes_types[index_e]) == 0:
                if index_b not in nb.setdefault(index_e, []):
                    nb[index_e].append(index_b)
        return nb

    def __repr__(self):
        n_atoms = int((self.nodes_types == 1).sum())
        return (f"<GraphStructure atoms={n_atoms} bonds={self.num_nodes - n_atoms} "
                f"edges={self.edge_index.shape[1]}>")


class GraphConstructor:
    def __init__(self, keypoints, image, config_dataset_keypoint, config_dataset_graph,
                 gt_molecular_graph=None, discarded_bond_length_factor=1.75,
                 restricted_proposals=True):
        self.config_dataset_keypoint = config_dataset_keypoint
        self.config_dataset_graph = config_dataset_graph
        self.max_number_neighbors = 6
        self.empty_test_window = [0.3, 0.3]
        self.discarded_bond_length_factor = discarded_bond_length_factor
        self.restricted_proposals = restricted_proposals

        self.symbols_classes_atoms, self.types_classes_bonds = load_vocabulary(
            config_dataset_graph["nb_atoms_classes"],
            config_dataset_graph["nb_bonds_classes"])
        self.atoms_classes_symbols = {v: k for k, v in self.symbols_classes_atoms.items()}
        self.bonds_classes_types = {v: k for k, v in self.types_classes_bonds.items()}

        self.image = image
        self.gt_molecular_graph = gt_molecular_graph

        if "mask_size" in self.config_dataset_keypoint:
            scaling_factor = (self.config_dataset_keypoint["image_size"][1]
                              / self.config_dataset_keypoint["mask_size"][1])
            self.keypoints = [
                [int(kp[0] * scaling_factor + (scaling_factor // 2)),
                 int(kp[1] * scaling_factor + (scaling_factor // 2))]
                for kp in keypoints]
        else:
            self.keypoints = keypoints

        self.bonds = []
        self.bonds_candidates = []
        self.bond_length = None
        self.test_window_size = (15, 15)
        if len(self.keypoints) >= 2:
            self.set_bonds(self.restricted_proposals)

    # ------------------------------------------------------------------ #
    def _bond_positions(self, bond):
        return ([self.keypoints[bond[0]][0], self.keypoints[bond[0]][1]],
                [self.keypoints[bond[1]][0], self.keypoints[bond[1]][1]])

    def _is_obstructed(self, bond):
        bond_positions = self._bond_positions(bond)
        bond_length = np.sqrt((bond_positions[0][0] - bond_positions[1][0]) ** 2
                              + (bond_positions[0][1] - bond_positions[1][1]) ** 2)
        test_window_size = (int(bond_length * 0.8), int(bond_length * 0.2))
        center = ((bond_positions[0][0] + bond_positions[1][0]) // 2,
                  (bond_positions[0][1] + bond_positions[1][1]) // 2)
        delta_x = bond_positions[0][0] - bond_positions[1][0]
        delta_y = bond_positions[0][1] - bond_positions[1][1]
        if delta_x == 0:
            angle = math.degrees(np.pi / 2)
        else:
            angle = math.degrees(atan(delta_y / delta_x))
        test_window = box_points(center, test_window_size, angle)
        for keypoint in self.keypoints:
            if point_in_polygon(keypoint[0], keypoint[1], test_window):
                return True
        return False

    def _is_too_long(self, bond):
        bond_positions = self._bond_positions(bond)
        length = np.sqrt((bond_positions[0][0] - bond_positions[1][0]) ** 2
                         + (bond_positions[0][1] - bond_positions[1][1]) ** 2)
        return length > self.discarded_bond_length_factor * self.bond_length

    def _is_empty_fast(self, bond):
        bond_positions = self._bond_positions(bond)
        center = ((bond_positions[0][0] + bond_positions[1][0]) // 2,
                  (bond_positions[0][1] + bond_positions[1][1]) // 2)
        image_crop = crop_pad(self.image,
                              top=center[1] - (self.test_window_size[1] // 2),
                              left=center[0] - (self.test_window_size[0] // 2),
                              height=self.test_window_size[1],
                              width=self.test_window_size[0])
        return bool((image_crop != 1.0).sum() < 1)

    # ------------------------------------------------------------------ #
    def set_candidates_bonds(self):
        min_distances = []
        for index_keypoint_query, keypoint_query in enumerate(self.keypoints):
            distances = []
            indices_keypoints_keys = []
            for index_keypoint_key, keypoint_key in enumerate(self.keypoints):
                if index_keypoint_key < index_keypoint_query:
                    distances.append(np.sqrt((keypoint_query[0] - keypoint_key[0]) ** 2
                                             + (keypoint_query[1] - keypoint_key[1]) ** 2))
                    indices_keypoints_keys.append(index_keypoint_key)
            if len(distances) > 0:
                min_distances.append(min(distances))
                indices_keypoint_neighbors = [
                    index_keypoint
                    for (distance, index_keypoint) in sorted(
                        zip(distances, indices_keypoints_keys), key=lambda x: x[0]
                    )[: self.max_number_neighbors]]
                for index_keypoint_neighbor in indices_keypoint_neighbors:
                    self.bonds_candidates.append([index_keypoint_query,
                                                  index_keypoint_neighbor])
        if len(min_distances) > 0:
            self.bond_length = np.percentile(min_distances, 75)
            self.test_window_size = (
                max(int(self.empty_test_window[0] * self.bond_length), 15),
                max(int(self.empty_test_window[1] * self.bond_length), 15))

    def set_bonds(self, restricted_proposals=True):
        self.set_candidates_bonds()
        for bond_candidate in self.bonds_candidates:
            if restricted_proposals and self._is_obstructed(bond_candidate):
                continue
            if restricted_proposals and self._is_empty_fast(bond_candidate):
                continue
            if restricted_proposals and self._is_too_long(bond_candidate):
                continue
            self.bonds.append(bond_candidate)

    def get_molecular_graph(self):
        """-> (atoms, bonds) dicts, positions in 1024x1024 image space."""
        atoms, bonds = [], []
        for index, keypoint in enumerate(self.keypoints):
            atoms.append({"index": index,
                          "class": self.symbols_classes_atoms["None"],
                          "position": keypoint, "type": 1})
        for bond in self.bonds:
            bonds.append({
                "index": bond,
                "class": self.types_classes_bonds["None"],
                "position": [(self.keypoints[bond[0]][0] + self.keypoints[bond[1]][0]) / 2,
                             (self.keypoints[bond[0]][1] + self.keypoints[bond[1]][1]) / 2],
                "type": 0})
        return atoms, bonds

    def build_structure(self):
        """-> :class:`GraphStructure` (atom nodes first, then bond nodes, both
        edge directions, exactly like ``set_nodes_edges_nodes_only`` +
        ``duplicate_edges`` + ``to_torch_nodes_only``)."""
        atoms, bonds = self.get_molecular_graph()
        nodes = [dict(a) for a in atoms]
        nb_atoms = len(atoms)
        edges = []
        for index, bond in enumerate(bonds):
            bond_node = dict(bond)
            bond_node["index"] = index + nb_atoms
            nodes.append(bond_node)
            b, e = bond["index"]
            edges.append([b, bond_node["index"]])
            edges.append([bond_node["index"], e])
        # duplicate_edges: add both directions of every edge
        all_edges = []
        for (u, v) in edges:
            all_edges.append([u, v])
            all_edges.append([v, u])
        edge_index = (torch.tensor(all_edges, dtype=torch.long).t().contiguous()
                      if all_edges else torch.zeros((2, 0), dtype=torch.long))
        return GraphStructure(
            edge_index=edge_index,
            nodes_positions=torch.tensor([n["position"] for n in nodes]).to(torch.long),
            nodes_classes=torch.tensor([n["class"] for n in nodes]),
            nodes_types=torch.tensor([n["type"] for n in nodes]),
        )
