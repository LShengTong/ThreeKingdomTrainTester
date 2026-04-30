from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, HeteroConv

from battle.environment.army import Army
from battle.environment.city import City


@dataclass
class GraphData:
    data: HeteroData
    army_id_to_index: Dict[int, int]
    city_id_to_index: Dict[int, int]


class BattleGCN(nn.Module):
    """Two-layer heterogeneous GCN over Army/City nodes."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.conv1 = HeteroConv(
            {
                ("Army", "target_city", "City"): GCNConv(input_dim, hidden_dim, add_self_loops=False),
                ("City", "targeted_by_army", "Army"): GCNConv(input_dim, hidden_dim, add_self_loops=False),
                ("Army", "target_army", "Army"): GCNConv(input_dim, hidden_dim),
            },
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {
                ("Army", "target_city", "City"): GCNConv(hidden_dim, output_dim, add_self_loops=False),
                ("City", "targeted_by_army", "Army"): GCNConv(hidden_dim, output_dim, add_self_loops=False),
                ("Army", "target_army", "Army"): GCNConv(hidden_dim, output_dim),
            },
            aggr="sum",
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {"Army": data["Army"].x, "City": data["City"].x}
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {node_type: F.relu(x) for node_type, x in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        return x_dict["Army"]


def build_battle_graph(cities: List[City], armies: List[Army]) -> GraphData:
    data = HeteroData()

    city_id_to_index = {city.get_city_id(): idx for idx, city in enumerate(cities)}
    army_id_to_index = {army.army_id: idx for idx, army in enumerate(armies)}

    city_features = [
        [float(city.get_pos()[0]), float(city.get_pos()[1]), float(city.soldier), 1.0]
        for city in cities
    ]
    army_features = [
        [float(army.get_pos()[0]), float(army.get_pos()[1]), float(army.soldier), 0.0]
        for army in armies
    ]

    data["City"].x = (
        torch.tensor(city_features, dtype=torch.float32)
        if city_features
        else torch.zeros((0, 4), dtype=torch.float32)
    )
    data["Army"].x = (
        torch.tensor(army_features, dtype=torch.float32)
        if army_features
        else torch.zeros((0, 4), dtype=torch.float32)
    )

    army_to_city_edges: List[List[int]] = []
    city_to_army_edges: List[List[int]] = []
    army_to_army_edges: List[List[int]] = []

    for src_army in armies:
        src_idx = army_id_to_index[src_army.army_id]
        target = src_army.target

        if isinstance(target, City):
            city_idx = city_id_to_index.get(target.get_city_id())
            if city_idx is None:
                continue
            army_to_city_edges.append([src_idx, city_idx])
            city_to_army_edges.append([city_idx, src_idx])
            continue

        if isinstance(target, Army):
            dst_idx = army_id_to_index.get(target.army_id)
            if dst_idx is None:
                continue
            army_to_army_edges.append([src_idx, dst_idx])

    data["Army", "target_city", "City"].edge_index = (
        torch.tensor(army_to_city_edges, dtype=torch.long).t().contiguous()
        if army_to_city_edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    data["City", "targeted_by_army", "Army"].edge_index = (
        torch.tensor(city_to_army_edges, dtype=torch.long).t().contiguous()
        if city_to_army_edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    data["Army", "target_army", "Army"].edge_index = (
        torch.tensor(army_to_army_edges, dtype=torch.long).t().contiguous()
        if army_to_army_edges
        else torch.zeros((2, 0), dtype=torch.long)
    )

    return GraphData(
        data=data,
        army_id_to_index=army_id_to_index,
        city_id_to_index=city_id_to_index,
    )
