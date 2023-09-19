import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from dotenv import dotenv_values
from torch_geometric.datasets import MoleculeNet

import gnnbuilder as gnnb
from gnnbuilder.code_gen import FPX

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(name)s][%(asctime)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("gnnb")

CURRENT_SCRIPT_DIR = Path(__file__).parent

env_config = dotenv_values("./.env")
if "BUILD_DIR" not in env_config:
    raise ValueError("BUILD_DIR not defined in env_config")
else:
    if env_config["BUILD_DIR"] is None:
        raise ValueError("BUILD_DIR not defined in env_config")
    else:
        build_dir_str = env_config["BUILD_DIR"]
        if not os.path.isdir(build_dir_str):
            raise ValueError(f"BUILD_DIR={build_dir_str} is not a valid path")
        else:
            BUILD_DIR = Path(build_dir_str)
if "VITIS_HLS_BIN" not in env_config:
    VITIS_HLS_BIN = "vitis_hls"
else:
    VITIS_HLS_BIN = env_config["VITIS_HLS_BIN"]
    if VITIS_HLS_BIN is None:
        VITIS_HLS_BIN = "vitis_hls"

torch.manual_seed(0)

dataset = MoleculeNet(root="./tmp/MoleculeNet", name="hiv").index_select(
    list(range(1000))
)
print(dataset)
print(f"dataset.num_classes={dataset.num_classes}")
print(f"dataset.num_features={dataset.num_features}")

dataset_max_node, dataset_max_edge = gnnb.compute_max_nodes_and_edges(dataset)
print(f"dataset_max_node={dataset_max_node}")
print(f"dataset_max_edge={dataset_max_edge}")

dataset_average_degree = gnnb.utils.compute_average_degree(dataset, round_val=True)
print(f"dataset_average_degree={dataset_average_degree}")

model = gnnb.GNNModel(
    graph_input_feature_dim=dataset.num_features,
    graph_input_edge_dim=dataset.num_edge_features,
    gnn_hidden_dim=16,
    gnn_num_layers=2,
    gnn_output_dim=8,
    gnn_conv=gnnb.SAGEConv_GNNB,
    gnn_activation=nn.ReLU,
    gnn_skip_connection=True,
    global_pooling=gnnb.GlobalPooling(["add", "mean", "max"]),
    mlp_head=gnnb.MLP(
        in_dim=8 * 3,
        out_dim=dataset.num_classes,
        hidden_dim=8,
        hidden_layers=3,
        activation=nn.ReLU,
        p_in=8,
        p_hidden=4,
        p_out=1,
    ),
    output_activation=None,
    gnn_p_in=1,
    gnn_p_hidden=8,
    gnn_p_out=4,
)

MAX_NODES = 600
MAX_EDGES = 600

num_nodes_guess, num_edges_guess = gnnb.compute_average_nodes_and_edges(
    dataset, round_val=True
)

PROJECT_NAME = "gnn_model"
VITIS_HLS_PATH = Path("/tools/software/xilinx/Vitis_HLS/2022.2/")
BUILD_DIR_PROJECT = BUILD_DIR / PROJECT_NAME

print(f"Project Name: {PROJECT_NAME}")
print(f"Vitis HLS Path: {VITIS_HLS_PATH}")
print(f"Build Directory: {BUILD_DIR_PROJECT}")


proj = gnnb.Project(
    "gnn_model",
    model,
    "classification_integer",
    VITIS_HLS_PATH,
    BUILD_DIR_PROJECT,
    dataset=dataset,
    max_nodes=MAX_NODES,
    max_edges=MAX_EDGES,
    num_nodes_guess=num_nodes_guess,
    num_edges_guess=num_edges_guess,
    degree_guess=dataset_average_degree,
    float_or_fixed="fixed",
    fpx=FPX(32, 16),
    fpga_part="xcu280-fsvh2892-2L-e",
    n_jobs=32,
)

proj.gen_hw_model()
proj.gen_testbench()
proj.gen_makefile()
proj.gen_vitis_hls_tcl_script()
proj.gen_makefile_vitis()

# tb_data = proj.build_and_run_testbench()
# print(tb_data)

# synth_data = proj.run_vitis_hls_synthesis()
# print(synth_data)
