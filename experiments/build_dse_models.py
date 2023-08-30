from functools import cache, partial
import json
from multiprocessing import Pool
import shutil
from typing import Any, Optional, Type, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import joblib

from pathlib import Path
import os
import itertools
import math
import random
# from pprint import pp

import torch.nn as nn

from torch_geometric.datasets import QM9

import gnnbuilder as gnnb
from gnnbuilder import MLP
from gnnbuilder.code_gen import FPX
from gnnbuilder.utils import compute_median_nodes_and_edges, compute_median_degree


VITIS_HLS_PATH = Path("/tools/software/xilinx/Vitis_HLS/2022.1/")


QM9_DATASET = QM9(root="./tmp/QM9").index_select(list(range(1000)))
DATASET_IN_DIM = QM9_DATASET.num_features
DATASET_OUT_DIM = QM9_DATASET[0].y.ravel().shape[0]

MEDIAN_NODES, MEDIAN_EDGES = compute_median_nodes_and_edges(QM9_DATASET, round_val=True)
MEDIAN_DEGREE = compute_median_degree(QM9_DATASET)

MAX_NODES = 600
MAX_EDGES = 600

CONVS = ["gcn", "gin", "pna", "sage"]
GNN_HIDDEN_DIM = [64, 128, 256]
GNN_OUT_DIM = [64, 128, 256]
GNN_NUM_LAYERS = [1, 2, 3, 4]
GNN_SKIP_CONNECTIONS = [True, False]
MLP_HIDDEN_DIM = [64, 128, 256]
MLP_NUM_LAYERS = [1, 2, 3, 4]

GNN_P_HIDDEN = [2, 4, 8]
GNN_P_OUT = [2, 4, 8]
MLP_P_IN = [2, 4, 8]
MLP_P_HIDDEN = [2, 4, 8]

VARS: list[list] = [
    CONVS,
    GNN_HIDDEN_DIM,
    GNN_OUT_DIM,
    GNN_NUM_LAYERS,
    GNN_SKIP_CONNECTIONS,
    MLP_HIDDEN_DIM,
    MLP_NUM_LAYERS,
    GNN_P_HIDDEN,
    GNN_P_OUT,
    MLP_P_IN,
    MLP_P_HIDDEN,
]
VAR_NAMES: list[str] = [
    "conv",
    "gnn_hidden_dim",
    "gnn_out_dim",
    "gnn_num_layers",
    "gnn_skip_connections",
    "mlp_hidden_dim",
    "mlp_num_layers",
    "gnn_p_hidden",
    "gnn_p_out",
    "mlp_p_in",
    "mlp_p_hidden",
]

ALL_COMBOS_LEN = math.prod([len(var) for var in VARS])
COMBO_TYPE = dict[str, Any]


def ALL_COMBOS():
    combos: list[COMBO_TYPE] = []
    for combo in tqdm.tqdm(itertools.product(*VARS), total=ALL_COMBOS_LEN):
        combos.append(dict(zip(VAR_NAMES, combo)))
    return combos


def gen_model_combos(seed: int = 0, num_models: int = 50):
    np.random.seed(seed)
    random.seed(seed)
    all_combos = ALL_COMBOS()
    sample_idxs = random.sample(range(ALL_COMBOS_LEN), num_models)
    sample_combos = [all_combos[idx] for idx in sample_idxs]
    return sample_combos, sample_idxs


def build_single_combo(
    combo: COMBO_TYPE, combo_idx: int, BUILD_DIR: Path, RESULTS_DIR: Path
):
    config_fp = RESULTS_DIR / f"perf_model_{combo_idx}_config.json"
    with open(config_fp, "w") as f:
        f.write(json.dumps(combo, indent=4))

    print(f"Building combo {combo_idx}")

    conv: Type
    if combo["conv"] == "gcn":
        conv = gnnb.GCNConv_GNNB
    elif combo["conv"] == "gin":
        conv = gnnb.GINConv_GNNB
    elif combo["conv"] == "pna":
        conv = gnnb.PNAConv_GNNB
    elif combo["conv"] == "sage":
        conv = gnnb.SAGEConv_GNNB
    else:
        raise ValueError(f"Unsupported conv: {combo['conv']}")

    model = gnnb.GNNModel(
        graph_input_feature_dim=DATASET_IN_DIM,
        graph_input_edge_dim=0,
        gnn_hidden_dim=combo["gnn_hidden_dim"],
        gnn_num_layers=combo["gnn_num_layers"],
        gnn_output_dim=combo["gnn_out_dim"],
        gnn_conv=conv,
        gnn_activation=nn.ReLU,
        gnn_skip_connection=combo["gnn_skip_connections"],
        global_pooling=gnnb.GlobalPooling(["add", "mean", "max"]),
        mlp_head=MLP(
            in_dim=combo["gnn_out_dim"] * 3,
            out_dim=DATASET_OUT_DIM,
            hidden_dim=combo["mlp_hidden_dim"],
            hidden_layers=combo["mlp_num_layers"],
            activation=nn.ReLU,
            p_in=combo["mlp_p_in"],
            p_hidden=combo["mlp_p_hidden"],
            p_out=1,
        ),
        output_activation=None,
        gnn_p_in=1,
        gnn_p_hidden=combo["gnn_p_hidden"],
        gnn_p_out=combo["gnn_p_out"],
    )

    project_name = f"perf_model_{combo_idx}_proj"

    proj = gnnb.Project(
        project_name,
        model,
        "regression",
        VITIS_HLS_PATH,
        BUILD_DIR,
        dataset=QM9_DATASET,
        max_nodes=MAX_NODES,
        max_edges=MAX_EDGES,
        num_nodes_guess=MEDIAN_NODES,
        num_edges_guess=MEDIAN_EDGES,
        degree_guess=MEDIAN_DEGREE,
        float_or_fixed="fixed",
        fpx=FPX(16, 10),
        fpga_part="xcu280-fsvh2892-2L-e",
    )

    print(f"Building project: {project_name}")
    proj.gen_hw_model()

    proj.gen_vitis_hls_tcl_script()
    synth_data = proj.run_vitis_hls_synthesis()

    synth_data_fp = RESULTS_DIR / f"perf_model_{combo_idx}_synth_data.json"
    with open(synth_data_fp, "w") as f:
        f.write(json.dumps(synth_data, indent=4))

    # delete the .autopilot folder
    proj_fp = BUILD_DIR / f"perf_model_{combo_idx}_proj"
    proj_autopilot_fp = (
        proj_fp
        / f"perf_model_{combo_idx}_proj_vitis_hls_project"
        / "solution1"
        / ".autopilot"
    )
    shutil.rmtree(proj_autopilot_fp)


def build_all_combos(
    combos, combos_idxs, BUILD_DIR: Path, PERF_DATA_V2_DIR: Path, n_jobs: int = 1
):
    # for combo, combo_idx in zip(combos, combos_idxs):
    #     build_single_combo(combo, combo_idx, BUILD_DIR)

    # use joblib to parallelize
    combo_pairs = list(zip(combos, combos_idxs))
    joblib.Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        joblib.delayed(build_single_combo)(
            combo, combo_idx, BUILD_DIR, PERF_DATA_V2_DIR
        )
        for combo, combo_idx in tqdm.tqdm(combo_pairs)
    )


if __name__ == "__main__":
    PERF_DATA_DIR = Path("./results_perf")
    os.makedirs(PERF_DATA_DIR, exist_ok=True)

    PERF_DATA_BUILD_DIR = Path("/usr/scratch/skaram7/gnnb_perf_builds/")
    os.makedirs(PERF_DATA_BUILD_DIR, exist_ok=True)

    combos, combos_idxs = gen_model_combos(seed=0, num_models=400)
    build_all_combos(combos, combos_idxs, PERF_DATA_BUILD_DIR, PERF_DATA_DIR, n_jobs=50)
