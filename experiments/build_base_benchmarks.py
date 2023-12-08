import itertools
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import tqdm
from torch_geometric.datasets import QM9, MoleculeNet
from torch_geometric.loader import DataLoader

import gnnbuilder as gnnb
from gnnbuilder.models import MLP


def compute_max_nodes_and_edges(dataset):
    max_node = 0
    max_edge = 0
    for data in dataset:
        max_node = max(max_node, data.num_nodes)
        max_edge = max(max_edge, data.num_edges)
    return max_node, max_edge


def compute_average_nodes_and_edges(dataset, round_val=True):
    avg_nodes = 0
    avg_edges = 0
    count = 0
    for data in dataset:
        avg_nodes += data.num_nodes
        avg_edges += data.num_edges
        count += 1
    avg_nodes /= count
    avg_edges /= count
    if round_val:
        avg_nodes = int(round(avg_nodes))
        avg_edges = round(avg_edges)
    return avg_nodes, avg_edges


def compute_median_nodes_and_edges(dataset):
    nodes = []
    edges = []
    for data in dataset:
        nodes.append(data.num_nodes)
        edges.append(data.num_edges)
    median_nodes = int(np.median(nodes))
    median_edges = int(np.median(edges))

    return median_nodes, median_edges


def round_up_to_nearest_multiple(num, multiple):
    return multiple * ((num + multiple - 1) // multiple)


def build_model(dim_in, dim_out, conv, parallel: bool = False):
    if not parallel:
        model = gnnb.GNNModel(
            graph_input_feature_dim=dim_in,
            graph_input_edge_dim=0,
            gnn_hidden_dim=128,
            gnn_num_layers=6,
            gnn_output_dim=64,
            gnn_conv=conv,
            gnn_activation=nn.ReLU,
            gnn_skip_connection=True,
            global_pooling=gnnb.GlobalPooling(["add", "mean", "max"]),
            mlp_head=MLP(
                in_dim=64 * 3,
                out_dim=dim_out,
                hidden_dim=64,
                hidden_layers=4,
                activation=nn.ReLU,
            ),
            output_activation=None,
        )
    elif parallel:
        if conv == gnnb.PNAConv_GNNB:
            gnn_p_hidden = 8
            gnn_p_out = 8
        else:
            gnn_p_hidden = 16
            gnn_p_out = 8

        # gnn_p_hidden = 16
        # gnn_p_out = 8

        model = gnnb.GNNModel(
            graph_input_feature_dim=dim_in,
            graph_input_edge_dim=0,
            gnn_hidden_dim=128,
            gnn_num_layers=6,
            gnn_output_dim=64,
            gnn_conv=conv,
            gnn_activation=nn.ReLU,
            gnn_skip_connection=True,
            global_pooling=gnnb.GlobalPooling(["add", "mean", "max"]),
            mlp_head=MLP(
                in_dim=64 * 3,
                out_dim=dim_out,
                hidden_dim=64,
                hidden_layers=4,
                activation=nn.ReLU,
                p_in=8,
                p_hidden=8,
                p_out=1,
            ),
            output_activation=None,
            gnn_p_in=1,
            gnn_p_hidden=gnn_p_hidden,
            gnn_p_out=gnn_p_out,
        )
    else:
        raise ValueError("parallel must be True or False")
    return model


CONVS = {
    "gcn": gnnb.GCNConv_GNNB,
    "gin": gnnb.GINConv_GNNB,
    "pna": gnnb.PNAConv_GNNB,
    "sage": gnnb.SAGEConv_GNNB,
}

DATASETS = {
    "qm9": QM9(root="./tmp/QM9").index_select(list(range(1000))),
    "esol": MoleculeNet(root="./tmp/MoleculeNet", name="esol").index_select(
        list(range(1000))
    ),
    "freesolv": MoleculeNet(root="./tmp/MoleculeNet", name="freesolv"),
    "lipo": MoleculeNet(root="./tmp/MoleculeNet", name="lipo").index_select(
        list(range(1000))
    ),
    "hiv": MoleculeNet(root="./tmp/MoleculeNet", name="hiv").index_select(
        list(range(1000))
    ),
}

DATASET_TASK_TYPE = {
    "qm9": "regression",
    "esol": "regression",
    "freesolv": "regression",
    "lipo": "regression",
    "hiv": "classification",
}


combo_names = itertools.product(CONVS.keys(), DATASETS.keys())
combo_values = itertools.product(CONVS.values(), DATASETS.values())
combos = list(zip(combo_names, combo_values))


def compute_pyg_cpu_benchmark(
    combos: list,
    RESULTS_DIR: Path,
    trials: int = 1,
    batch_size: int = 1,
    cpu_idx: int = 0,
):
    for combo in combos:
        names, objects = combo
        conv_name, dataset_name = names
        conv, dataset = objects
        print(f"{conv_name}: {conv}")
        print(f"{dataset_name}: {dataset}")

        num_features = dataset.num_features
        print(f"num_features: {num_features}")

        task_type = DATASET_TASK_TYPE[dataset_name]
        print(f"task_type: {task_type}")
        if task_type == "classification":
            dim_out = dataset.num_classes
        else:
            dim_out = dataset[0].y.ravel().shape[0]

        runtimes: list[float] = []

        model = build_model(dataset.num_features, dim_out, conv)
        model = model.to("cpu")
        model.eval()

        old_cpu_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {cpu_idx})

        total_energy = 0.0
        total_time = 0.0

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in tqdm.tqdm(dataloader):
            batch = batch.to("cpu")
            x = batch.x.type(torch.float32).to("cpu")
            edge_index = batch.edge_index.type(torch.long).to("cpu")
            timer = benchmark.Timer(
                stmt="model(x, edge_index)",
                globals={"x": x, "edge_index": edge_index, "model": model},
            )

            with open("/sys/class/powercap/intel-rapl:1/energy_uj") as f:
                start_energy_uj = int(f.read())
                start_time = time.time()
            t = timer.timeit(trials).mean
            with open("/sys/class/powercap/intel-rapl:1/energy_uj") as f:
                end_energy_uj = int(f.read())
                end_time = time.time()

            total_energy += (end_energy_uj - start_energy_uj) / 1e6
            total_time += end_time - start_time

            t_per_graph = t / batch_size
            runtimes.append(t_per_graph)

        os.sched_setaffinity(0, old_cpu_affinity)

        average_runtime = np.mean(runtimes)
        average_power = total_energy / total_time

        data_path = (
            RESULTS_DIR
            / f"runtime_pyg_cpu_{conv_name}_{dataset_name}_b{batch_size}.txt"
        )
        with open(data_path, "w") as f:
            f.write(f"average_runtime {average_runtime}\n")
            for runtime in runtimes:
                f.write(f"{runtime}\n")

        data_path_energy = (
            RESULTS_DIR / f"energy_pyg_cpu_{conv_name}_{dataset_name}_b{batch_size}.txt"
        )
        with open(data_path_energy, "w") as f:
            f.write(f"average_power {average_power}\n")
            f.write(f"total_energy {total_energy}\n")
            f.write(f"total_time {total_time}\n")


def compute_pyg_gpu_benchmark(
    combos: list,
    RESULTS_DIR: Path,
    trials: int = 1,
    batch_size: int = 1,
    gpu_idx: int = 0,
):
    assert torch.cuda.is_available()

    device = torch.device(f"cuda:{gpu_idx}")

    for combo in combos:
        names, objects = combo
        conv_name, dataset_name = names
        conv, dataset = objects
        print(f"{conv_name}: {conv}")
        print(f"{dataset_name}: {dataset}")

        num_features = dataset.num_features
        print(f"num_features: {num_features}")

        task_type = DATASET_TASK_TYPE[dataset_name]
        print(f"task_type: {task_type}")
        if task_type == "classification":
            dim_out = dataset.num_classes
        else:
            dim_out = dataset[0].y.ravel().shape[0]

        runtimes: list[float] = []

        model = build_model(dataset.num_features, dim_out, conv)
        model = model.to(device)
        model.eval()

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        p = subprocess.Popen(
            [
                "nvidia-smi",
                "dmon",
                "-s",
                "p",
                "-d",
                "1",
                "-i",
                str(gpu_idx),
                "-o",
                "DT",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )

        for batch in tqdm.tqdm(dataloader):
            batch = batch.to(device)
            x = batch.x.type(torch.float32).to(device)
            edge_index = batch.edge_index.type(torch.long).to(device)
            timer = benchmark.Timer(
                stmt="model(x, edge_index)",
                globals={"x": x, "edge_index": edge_index, "model": model},
            )

            t = timer.timeit(trials).mean

            t_per_graph = t / batch_size
            runtimes.append(t_per_graph)

        p.terminate()

        if p.stdout:
            stdout = p.stdout.read()
            print(stdout)
        if p.stderr:
            stderr = p.stderr.read()
            if stderr:
                raise RuntimeError(stderr)

        # process data
        data_raw = stdout
        data_raw_no_header = "\n".join(
            [line for line in data_raw.split("\n") if not line.startswith("#")]
        )
        data_raw_split = [
            line.split() for line in data_raw_no_header.split("\n") if line
        ]
        power_df = pd.DataFrame(
            data_raw_split,
            columns=["date", "time", "gpu_idx", "power", "gtemp", "m_temp"],
        )
        # print(power_df)

        # combine date and time columns to pandas datetime
        power_df["datetime"] = pd.to_datetime(power_df["date"] + " " + power_df["time"])
        power_df["power"] = power_df["power"].astype(float)
        # print(power_df)

        # compute average power
        average_power = power_df["power"].mean()
        # compute total duration seconds
        total_time = (
            power_df["datetime"].iloc[-1] - power_df["datetime"].iloc[0]
        ).total_seconds()
        # compute total energy in Joules
        total_energy = power_df["power"].mean() * total_time

        average_runtime = np.mean(runtimes)
        data_path = (
            RESULTS_DIR
            / f"runtime_pyg_gpu_{conv_name}_{dataset_name}_b{batch_size}.txt"
        )
        with open(data_path, "w") as f:
            f.write(f"average_runtime {average_runtime}\n")
            for runtime in runtimes:
                f.write(f"{runtime}\n")

        data_path_energy = (
            RESULTS_DIR / f"energy_pyg_gpu_{conv_name}_{dataset_name}_b{batch_size}.txt"
        )
        with open(data_path_energy, "w") as f:
            f.write(f"average_power {average_power}\n")
            f.write(f"total_energy {total_energy}\n")
            f.write(f"total_time {total_time}\n")


if __name__ == "__main__":
    RESULTS_DIR = Path("./results_testing")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    compute_pyg_cpu_benchmark(combos, RESULTS_DIR, trials=5, batch_size=1, cpu_idx=0)
    compute_pyg_gpu_benchmark(combos, RESULTS_DIR, trials=5, batch_size=1, gpu_idx=2)
    compute_pyg_gpu_benchmark(combos, RESULTS_DIR, trials=5, batch_size=4, gpu_idx=2)
