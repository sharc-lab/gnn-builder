import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, ModuleList
from torch import Tensor

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    # GINEConv,
    GATConv,
    GATv2Conv,
    PNAConv,
    SAGEConv,
)
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import aggr
from torch_geometric.data import Data
from torch_geometric.typing import Adj

from typing import Union, Optional, Callable
from rich.pretty import pprint as pp

from .utils import layer_param_name_combiner

TorchModuleArg = Callable[..., torch.nn.Module]
TorchModuleArgOptional = Optional[Callable[..., torch.nn.Module]]


def compute_param_bits(p, bit_size: int = 32):
    return int(p.numel() * bit_size)


class GCNConv_GNNB(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, p_in: int = 1, p_out: int = 1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.p_in = p_in
        self.p_out = p_out

        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        return self.conv(x, edge_index)


class GIN_MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.linear_0 = nn.Linear(self.in_dim, self.hidden_dim)
        self.linear_1 = nn.Linear(self.hidden_dim, self.out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_0(x)
        x = self.relu(x)
        x = self.linear_1(x)
        return x


class GINConv_GNNB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 64,
        eps: float = 0.0,
        p_in: int = 1,
        p_out: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.eps = eps

        self.p_in = p_in
        self.p_out = p_out

        self.mlp = GIN_MLP(in_channels, out_channels, hidden_dim)
        self.conv = GINConv(self.mlp, eps=eps, train_eps=False)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        return self.conv(x, edge_index)


class GINEConv_GNNB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        hidden_dim: int = 64,
        eps: float = 0.0,
        p_in: int = 1,
        p_out: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.eps = eps

        self.p_in = p_in
        self.p_out = p_out

        self.mlp = GIN_MLP(in_channels, out_channels, hidden_dim)
        self.conv = GINEConv(self.mlp, eps=eps, train_eps=False, edge_dim=edge_dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return self.conv.forward(x, edge_index, edge_attr=edge_attr)


class GATConv_GNNB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        p_in: int = 1,
        p_out: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.bias = bias

        self.p_in = p_in
        self.p_out = p_out

        self.conv = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        return self.conv(x, edge_index)


class GATEdgeConv_GNNB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        p_in: int = 1,
        p_out: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.bias = bias

        self.p_in = p_in
        self.p_out = p_out

        self.conv = GATConv(
            in_channels,
            out_channels,
            edge_dim=edge_dim,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return self.conv.forward(x, edge_index, edge_attr=edge_attr)


class PNAConv_GNNB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        delta: float = 1.0,
        p_in: int = 1,
        p_out: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.delta = delta

        self.p_in = p_in
        self.p_out = p_out

        self.aggregators = ["max", "min", "mean", "std"]
        self.scalers = ["identity", "amplification", "attenuation"]

        fake_deg_tensor = torch.tensor([0, 1])

        self.conv = PNAConv(
            in_channels, out_channels, self.aggregators, self.scalers, fake_deg_tensor
        )
        self.conv.aggr_module.init_avg_deg_log = self.delta
        self.conv.aggr_module.reset_parameters()

        self.delta_scaler = self.conv.aggr_module.avg_deg_log.item()

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        return self.conv(x, edge_index)


class SAGEConv_GNNB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        p_in: int = 1,
        p_out: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.p_in = p_in
        self.p_out = p_out

        self.conv = SAGEConv(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        return self.conv(x, edge_index)


SUPPORTED_GLOBAL_POOLING_AGGRS = {
    "add": "SumAggregation",
    "max": "MaxAggregation",
    "mean": "MeanAggregation",
}

SUPPORTED_GLOBAL_POOLING_MODE = ["cat"]


class GlobalPooling(torch.nn.Module):
    def __init__(self, aggrs: list[str], mode: str = "cat"):
        super().__init__()
        self.aggrs = aggrs
        self.mode = mode

        if aggrs == []:
            raise ValueError("Aggregation list is empty.")

        for aggr_str in self.aggrs:
            if aggr_str not in SUPPORTED_GLOBAL_POOLING_AGGRS:
                raise NotImplementedError(
                    f"Aggregation {aggr_str} is not supported. Supported aggregations are {SUPPORTED_GLOBAL_POOLING_AGGRS}."
                )

        if self.mode not in SUPPORTED_GLOBAL_POOLING_MODE:
            raise NotImplementedError(
                f"Mode {self.mode} is not supported. Supported modes are {SUPPORTED_GLOBAL_POOLING_MODE}."
            )

        aggrs_modules = [SUPPORTED_GLOBAL_POOLING_AGGRS[aggr] for aggr in self.aggrs]
        self.multi_aggr = aggr.MultiAggregation(aggrs_modules, mode=self.mode)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.multi_aggr(*args, **kwargs)

    @property
    def num_of_aggrs(self) -> int:
        return len(self.aggrs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.aggrs}, mode={self.mode})"


SUPPORTED_ACTIVATIONS = [nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh]


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        activation: TorchModuleArg = nn.ReLU,
        norm_layer: TorchModuleArgOptional = None,
        p_in: int = 1,
        p_hidden: int = 1,
        p_out: int = 1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.norm_layer = norm_layer

        # check to see if activation is supported
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"activation {activation} not supported")

        # norm not supported yet
        if self.norm_layer is not None:
            raise NotImplementedError("norm not supported yet")

        self.p_in = p_in
        self.p_hidden = p_hidden
        self.p_out = p_out

        self.linear_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        if hidden_layers == 0:
            self.linear_layers.append(nn.Linear(self.in_dim, self.out_dim))
        if hidden_layers > 0:
            for i in range(hidden_layers):
                if i == 0:
                    linear_in_dim = self.in_dim
                else:
                    linear_in_dim = self.hidden_dim
                self.linear_layers.append(nn.Linear(linear_in_dim, self.hidden_dim))
                if self.norm_layer is not None:
                    self.norm_layers.append(self.norm_layer(self.hidden_dim))
                if self.activation is not None:
                    self.activations.append(self.activation())
            self.linear_layers.append(nn.Linear(self.hidden_dim, self.out_dim))

        self.layer_list = []
        for i in range(len(self.linear_layers)):
            if i < len(self.linear_layers) - 1:
                self.layer_list.append(self.linear_layers[i])
                if self.norm_layer is not None:
                    self.layer_list.append(self.norm_layers[i])
                if self.activation is not None:
                    self.layer_list.append(self.activations[i])
            else:
                self.layer_list.append(self.linear_layers[i])
        self.mlp = nn.Sequential(*self.layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    @property
    def p_factors(self) -> list[tuple[int, int]]:
        p_factors: list[tuple[int, int]] = []
        if self.hidden_layers == 0:
            p_factors.append((self.p_in, self.p_out))
        elif self.hidden_layers > 0:
            for i in range(self.hidden_layers):
                if i == 0:
                    p_factors.append((self.p_in, self.p_hidden))
                else:
                    p_factors.append((self.p_hidden, self.p_hidden))
            p_factors.append((self.p_hidden, self.p_out))
        else:
            raise ValueError("hidden_layers must be >= 0")
        return p_factors
    
    @property
    def num_of_layers(self) -> int:
        return len(self.linear_layers)


SUPPORTED_GNN_CONVS = [
    GCNConv_GNNB,
    GINConv_GNNB,
    GATConv_GNNB,
    PNAConv_GNNB,
    SAGEConv_GNNB,
]


class GNNModel(nn.Module):
    def __init__(
        self,
        graph_input_feature_dim: int,
        graph_input_edge_dim: Optional[int],
        gnn_hidden_dim: int,
        gnn_num_layers: int,
        gnn_output_dim: int,
        gnn_conv: TorchModuleArg,
        gnn_activation: TorchModuleArg,
        gnn_skip_connection: bool,
        global_pooling: GlobalPooling,
        mlp_head: MLP,
        output_activation: TorchModuleArgOptional,
        gnn_p_in: int = 1,
        gnn_p_hidden: int = 1,
        gnn_p_out: int = 1,
    ) -> None:
        super().__init__()

        self.graph_input_feature_dim = graph_input_feature_dim
        self.graph_input_edge_dim = graph_input_edge_dim

        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_num_layers = gnn_num_layers
        self.gnn_output_dim = gnn_output_dim
        self.gnn_conv = gnn_conv
        if self.gnn_conv not in SUPPORTED_GNN_CONVS:
            raise ValueError(f"gnn_conv must be one of {SUPPORTED_GNN_CONVS}")
        self.gnn_activation = gnn_activation
        if self.gnn_activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"gnn_activation must be one of {SUPPORTED_ACTIVATIONS}")
        self.gnn_skip_connection = gnn_skip_connection

        self.global_pooling = global_pooling
        self.mlp_head = mlp_head
        self.output_activation = output_activation

        self.output_activation_module = None
        if self.output_activation is not None:
            self.output_activation_module = self.output_activation(dim=-1)

        self.gnn_p_in = gnn_p_in
        self.gnn_p_hidden = gnn_p_hidden
        self.gnn_p_out = gnn_p_out

        # Build GNN head
        # TODO: refactor this into a sperate GNNHead class
        self.gnn_convs = ModuleList()
        self.gnn_activations = ModuleList()
        if self.gnn_num_layers == 0:
            if self.graph_input_feature_dim != self.gnn_output_dim:
                raise ValueError(
                    f"You specified gnn_num_layers=0, but (gnn_output_dim={self.gnn_output_dim}) != (graph_input_feature_dim={self.graph_input_feature_dim})."
                )
        if self.gnn_num_layers == 1:
            self.gnn_convs.append(
                self.gnn_conv(
                    self.graph_input_feature_dim,
                    self.gnn_output_dim,
                    p_in=self.gnn_p_in,
                    p_out=self.gnn_p_out,
                )
            )
            self.gnn_activations.append(self.gnn_activation())
        if self.gnn_num_layers > 1:
            for i in range(self.gnn_num_layers):
                if i == 0:
                    in_dim = self.graph_input_feature_dim
                    out_dim = self.gnn_hidden_dim
                    p_in = self.gnn_p_in
                    p_out = self.gnn_p_hidden
                elif i == self.gnn_num_layers - 1:
                    in_dim = self.gnn_hidden_dim
                    out_dim = self.gnn_output_dim
                    p_in = self.gnn_p_hidden
                    p_out = self.gnn_p_out
                else:
                    in_dim = self.gnn_hidden_dim
                    out_dim = self.gnn_hidden_dim
                    p_in = self.gnn_p_hidden
                    p_out = self.gnn_p_hidden
                self.gnn_convs.append(
                    self.gnn_conv(in_dim, out_dim, p_in=p_in, p_out=p_out)
                )
                self.gnn_activations.append(self.gnn_activation())

    def forward(
        self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None
    ) -> Tensor:
        x_gnn = x

        if self.gnn_num_layers == 0:
            x_gnn = x_gnn
        if self.gnn_num_layers >= 1:
            for i, (conv, act) in enumerate(zip(self.gnn_convs, self.gnn_activations)):
                x_gnn_in = x_gnn
                x_gnn = conv(x_gnn, edge_index)
                if self.gnn_skip_connection:
                    if (i != 0) and (i != self.gnn_num_layers - 1):
                        x_gnn = x_gnn + x_gnn_in
                if act is not None:
                    x_gnn = act(x_gnn)

        x_gnn_out = x_gnn
        x_gnn_global_pooling_out = self.global_pooling(x_gnn_out)

        x_mlp_out = self.mlp_head(x_gnn_global_pooling_out)
        if self.output_activation_module is not None:
            x_mlp_out = self.output_activation_module(x_mlp_out)

        return x_mlp_out

    @property
    def input_node_features_dim(self):
        return self.graph_input_feature_dim

    @property
    def input_edge_features_dim(self):
        return self.graph_input_edge_dim

    @property
    def output_features_dim(self):
        return self.mlp_head.out_dim

    @property
    def gnn_layer_sizes(self):
        return [
            (in_dim, out_dim)
            for in_dim, out_dim in map(
                lambda x: (x.in_channels, x.out_channels), self.gnn_convs
            )
        ]

    @property
    def layers(self):
        return dict(self.named_children())

    @property
    def layer_names(self):
        # return {k: f"{v.__class__.__name__}_{k}" for k, v in self.layers.items()}
        return {k: f"{k}" for k, v in self.layers.items()}  # this is dumb but im lazy

    @property
    def layer_parameters(self):
        return {k: list(v.named_parameters()) for k, v in self.layers.items()}

    @property
    def layer_parameters_flat(self):
        return [p for l in self.layer_parameters.values() for p in l]

    @property
    def layer_parameter_names(self):
        return {
            k: [layer_param_name_combiner(self.layer_names[k], p[0]) for p in v]
            for k, v in self.layer_parameters.items()
        }

    @property
    def layer_parameter_names_flat(self):
        return [p for l in self.layer_parameter_names.values() for p in l]

    @property
    def layer_parameter_shapes(self):
        return {
            k: [list(p[1].size()) for p in v] for k, v in self.layer_parameters.items()
        }

    @property
    def layer_parameter_shapes_flat(self):
        return [p for l in self.layer_parameter_shapes.values() for p in l]
