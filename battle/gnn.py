from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from battle.army import Army
from battle.city import City


@dataclass
class GraphData:
    node_features: torch.Tensor
    edge_index: torch.Tensor

class TreeLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W_self   = nn.Linear(in_features, out_features)
        self.W_parent = nn.Linear(in_features, out_features)

    def forward(self, h, parent):
        return torch.relu(
            self.W_self(h) + self.W_parent(parent)
        )


class BattleGCN(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: List[GCNLayer] = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(GCNLayer(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = node_features
        for layer in self.layers:
            h = layer(h, edge_index)
        return h


def build_battle_graph(
    cities: List[City],
    armies: List[Army],
    attack_radius: float,
) -> GraphData:
    # Node feature: [x, y, soldier, is_city]
    city_features = [
        torch.tensor([c.get_pos()[0], c.get_pos()[1], float(c.soldier), 1.0])
        for c in cities
    ]
    army_features = [
        torch.tensor([a.get_pos()[0], a.get_pos()[1], float(a.soldier), 0.0])
        for a in armies
    ]
    features = city_features + army_features

    if len(features) == 0:
        return GraphData(
            node_features=torch.zeros((0, 4), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )

    node_features = torch.stack(features).float()

    edges = []
    num_nodes = node_features.size(0)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            dist = torch.dist(node_features[i, :2], node_features[j, :2]).item()
            if dist <= attack_radius:
                edges.append((i, j))

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return GraphData(node_features=node_features, edge_index=edge_index)
