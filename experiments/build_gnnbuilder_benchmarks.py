import joblib
import numpy as np
import tqdm

from pathlib import Path
import os
import itertools

# from pprint import pp

import torch.nn as nn
from torch_geometric.datasets import MoleculeNet, QM9

import gnnbuilder as gnnb
from gnnbuilder.code_gen import FPX
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


def compute_cpp_cpu_benchmark(
    combos: list,
    RESULTS_DIR: Path,
    BUILD_DIR: Path,
    VITIS_HLS_PATH: Path,
    n_jobs: int = 32,
):
    MAX_NODES = 600
    MAX_EDGES = 600

    def compute_runtime_single(combo):
        names, objects = combo
        conv_name, dataset_name = names
        conv, dataset = objects
        print(f"{conv_name}: {conv}")
        print(f"{dataset_name}: {dataset}")

        median_nodes, median_edges = compute_median_nodes_and_edges(dataset)
        print(f"median_nodes: {median_nodes}")
        print(f"median_edges: {median_edges}")

        median_degree = gnnb.utils.compute_median_degree(dataset)
        print(f"median_degree: {median_degree}")

        num_features = dataset.num_features
        print(f"num_features: {num_features}")

        task_type = DATASET_TASK_TYPE[dataset_name]
        print(f"task_type: {task_type}")
        if task_type == "classification":
            dim_out = dataset.num_classes
        else:
            dim_out = dataset[0].y.ravel().shape[0]
        print(f"dim_out: {dim_out}")

        if task_type == "classification":
            output_encoding = "classification_integer"
        elif task_type == "regression":
            output_encoding = "regression"
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        model = build_model(dataset.num_features, dim_out, conv)

        project_name = f"cpp_cpu_{conv_name}_{dataset_name}"

        proj = gnnb.Project(
            project_name,
            model,
            output_encoding,
            VITIS_HLS_PATH,
            BUILD_DIR,
            dataset=dataset,
            max_nodes=MAX_NODES,
            max_edges=MAX_EDGES,
            num_nodes_guess=median_nodes,
            num_edges_guess=median_edges,
            degree_guess=median_degree,
            float_or_fixed="float",
            fpga_part="xcu280-fsvh2892-2L-e",
        )

        proj.gen_hw_model()
        proj.gen_testbench()
        proj.gen_makefile()
        tb_data = proj.build_and_run_testbench()

        average_runtime = tb_data["model_runtime"]
        data_path = RESULTS_DIR / f"runtime_cpp_cpu_{conv_name}_{dataset_name}.txt"
        with open(data_path, "w") as f:
            f.write(f"average_runtime {average_runtime}\n")

    joblib.Parallel(n_jobs=n_jobs, backend="loky")(
        joblib.delayed(compute_runtime_single)(combo) for combo in tqdm.tqdm(combos)
    )
    # for combo in tqdm.tqdm(combos):
    #     compute_runtime_single(combo)


def compute_fpga_base_runtime_single(
    combo,
    RESULTS_DIR: Path,
    BUILD_DIR: Path,
    VITIS_HLS_PATH: Path,
    MAX_NODES: int = 600,
    MAX_EDGES: int = 600,
):
    names, objects = combo
    conv_name, dataset_name = names
    conv, dataset = objects
    print(f"{conv_name}: {conv}")
    print(f"{dataset_name}: {dataset}")

    median_nodes, median_edges = compute_median_nodes_and_edges(dataset)
    print(f"median_nodes: {median_nodes}")
    print(f"median_edges: {median_edges}")

    median_degree = gnnb.utils.compute_median_degree(dataset)
    print(f"median_degree: {median_degree}")

    num_features = dataset.num_features
    print(f"num_features: {num_features}")

    task_type = DATASET_TASK_TYPE[dataset_name]
    print(f"task_type: {task_type}")
    if task_type == "classification":
        dim_out = dataset.num_classes
    else:
        dim_out = dataset[0].y.ravel().shape[0]
    print(f"dim_out: {dim_out}")

    if task_type == "classification":
        output_encoding = "classification_integer"
    elif task_type == "regression":
        output_encoding = "regression"
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    model = build_model(dataset.num_features, dim_out, conv, parallel=False)

    project_name = f"fpga_base_{conv_name}_{dataset_name}"

    proj = gnnb.Project(
        project_name,
        model,
        output_encoding,
        VITIS_HLS_PATH,
        BUILD_DIR,
        dataset=dataset,
        max_nodes=MAX_NODES,
        max_edges=MAX_EDGES,
        num_nodes_guess=median_nodes,
        num_edges_guess=median_edges,
        degree_guess=median_degree,
        float_or_fixed="fixed",
        fpx=FPX(32, 16),
        fpga_part="xcu280-fsvh2892-2L-e",
    )

    proj.gen_hw_model()
    proj.gen_testbench(gen_testbench_data=False)
    proj.gen_makefile()
    proj.gen_vitis_hls_tcl_script()

    synth_data = proj.run_vitis_hls_synthesis()

    average_runtime = synth_data["worst_case_runtime"]
    print(f"average_runtime: {average_runtime}")
    data_path = RESULTS_DIR / f"runtime_fpga_base_{conv_name}_{dataset_name}.txt"
    with open(data_path, "w") as f:
        f.write(f"average_runtime {average_runtime}\n")

    resource_data_path = (
        RESULTS_DIR / f"resources_fpga_base_{conv_name}_{dataset_name}.txt"
    )
    with open(resource_data_path, "w") as f:
        f.write(f"bram {synth_data['bram']}\n")
        f.write(f"dsp {synth_data['dsp']}\n")
        f.write(f"ff {synth_data['ff']}\n")
        f.write(f"lut {synth_data['lut']}\n")
        f.write(f"uram {synth_data['uram']}\n")


def compute_fpga_base_benchmark(
    combos: list,
    RESULTS_DIR: Path,
    BUILD_DIR: Path,
    VITIS_HLS_PATH: Path,
    n_jobs: int = 32,
):
    MAX_NODES = 600
    MAX_EDGES = 600

    #  use joblib to parallelize the computation
    # joblib.Parallel(n_jobs=n_jobs, backend="multiprocessing")(joblib.delayed(compute_runtime_single)(combo) for combo in tqdm.tqdm(combos))
    # for combo in combos:
    #     compute_runtime_single(combo)

    joblib.Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        joblib.delayed(compute_fpga_base_runtime_single)(
            combo, RESULTS_DIR, BUILD_DIR, VITIS_HLS_PATH, MAX_NODES, MAX_EDGES
        )
        for combo in tqdm.tqdm(combos)
    )


def compute_fpga_par_runtime_single(
    combo,
    RESULTS_DIR: Path,
    BUILD_DIR: Path,
    VITIS_HLS_PATH: Path,
    MAX_NODES: int = 600,
    MAX_EDGES: int = 600,
):
    names, objects = combo
    conv_name, dataset_name = names
    conv, dataset = objects
    print(f"{conv_name}: {conv}")
    print(f"{dataset_name}: {dataset}")

    median_nodes, median_edges = compute_median_nodes_and_edges(dataset)
    print(f"median_nodes: {median_nodes}")
    print(f"median_edges: {median_edges}")

    median_degree = gnnb.utils.compute_median_degree(dataset)
    print(f"median_degree: {median_degree}")

    num_features = dataset.num_features
    print(f"num_features: {num_features}")

    task_type = DATASET_TASK_TYPE[dataset_name]
    print(f"task_type: {task_type}")
    if task_type == "classification":
        dim_out = dataset.num_classes
    else:
        dim_out = dataset[0].y.ravel().shape[0]
    print(f"dim_out: {dim_out}")

    if task_type == "classification":
        output_encoding = "classification_integer"
    elif task_type == "regression":
        output_encoding = "regression"
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    model = build_model(dataset.num_features, dim_out, conv, parallel=True)

    project_name = f"fpga_par_{conv_name}_{dataset_name}"

    proj = gnnb.Project(
        project_name,
        model,
        output_encoding,
        VITIS_HLS_PATH,
        BUILD_DIR,
        dataset=dataset,
        max_nodes=MAX_NODES,
        max_edges=MAX_EDGES,
        num_nodes_guess=median_nodes,
        num_edges_guess=median_edges,
        degree_guess=median_degree,
        float_or_fixed="fixed",
        fpx=FPX(16, 10),
        fpga_part="xcu280-fsvh2892-2L-e",
    )

    print(proj.model_dir)

    proj.gen_hw_model()
    proj.gen_testbench(gen_testbench_data=False)
    proj.gen_makefile()
    proj.gen_vitis_hls_tcl_script()

    synth_data = proj.run_vitis_hls_synthesis()

    average_runtime = synth_data["worst_case_runtime"]
    print(f"average_runtime: {average_runtime}")
    data_path = RESULTS_DIR / f"runtime_fpga_par_{conv_name}_{dataset_name}.txt"
    with open(data_path, "w") as f:
        f.write(f"average_runtime {average_runtime}\n")

    resource_data_path = (
        RESULTS_DIR / f"resources_fpga_par_{conv_name}_{dataset_name}.txt"
    )
    with open(resource_data_path, "w") as f:
        f.write(f"bram {synth_data['bram']}\n")
        f.write(f"dsp {synth_data['dsp']}\n")
        f.write(f"ff {synth_data['ff']}\n")
        f.write(f"lut {synth_data['lut']}\n")
        f.write(f"uram {synth_data['uram']}\n")


def compute_fpga_par_benchmark(
    combos: list,
    RESULTS_DIR: Path,
    BUILD_DIR: Path,
    VITIS_HLS_PATH: Path,
    n_jobs: int = 32,
):
    MAX_NODES = 600
    MAX_EDGES = 600

    joblib.Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        joblib.delayed(compute_fpga_par_runtime_single)(
            combo, RESULTS_DIR, BUILD_DIR, VITIS_HLS_PATH, MAX_NODES, MAX_EDGES
        )
        for combo in tqdm.tqdm(combos)
    )

    #  use joblib to parallelize the computation
    # joblib.Parallel(n_jobs=n_jobs, backend="threading")(joblib.delayed(compute_runtime_single)(combo) for combo in tqdm.tqdm(combos))
    # for combo in combos[5:]:
    #     compute_runtime_single(combo)
    # if combo[0] == ('pna', 'qm9'):
    #     print(combo)
    #     compute_runtime_single(combo)


# def gather_resource_data(combos: list, RESULTS_DIR: Path, BUILD_DIR: Path):
#     for combo in combos:
#         names, objects = combo
#         conv_name, dataset_name = names
#         conv, dataset = objects
#         print(f"{conv_name}: {conv}")
#         print(f"{dataset_name}: {dataset}")

#         proj_fp = BUILD_DIR / f"fpga_base_{conv_name}_{dataset_name}"
#         proj_synth_rpt_fp = (
#             proj_fp
#             / f"fpga_base_{conv_name}_{dataset_name}_vitis_hls_project"
#             / "solution1"
#             / "syn"
#             / "report"
#             / f"fpga_base_{conv_name}_{dataset_name}_top_csynth.xml"
#         )

#         synth_data = gnnb.code_gen.extract_data_from_csynth_report(proj_synth_rpt_fp)
#         # pp(synth_data)
#         # exit()

#         average_runtime = synth_data["worst_case_runtime"]
#         print(f"average_runtime: {average_runtime}")
#         data_path = RESULTS_DIR / f"runtime_fpga_base_{conv_name}_{dataset_name}.txt"
#         with open(data_path, "w") as f:
#             f.write(f"average_runtime {average_runtime}\n")

#         resource_data_path = (
#             RESULTS_DIR / f"resources_fpga_base_{conv_name}_{dataset_name}.txt"
#         )
#         with open(resource_data_path, "w") as f:
#             f.write(f"bram {synth_data['bram']}\n")
#             f.write(f"dsp {synth_data['dsp']}\n")
#             f.write(f"ff {synth_data['ff']}\n")
#             f.write(f"lut {synth_data['lut']}\n")
#             f.write(f"uram {synth_data['uram']}\n")


if __name__ == "__main__":
    RESULTS_DIR = Path("./results_gnnb/")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    BUILD_DIR = Path("/usr/scratch/skaram7/gnnb_main_builds/")
    os.makedirs(BUILD_DIR, exist_ok=True)

    VITIS_HLS_PATH = Path("/tools/software/xilinx/Vitis_HLS/2022.2/")

    # compute_cpp_cpu_benchmark(combos, RESULTS_DIR, BUILD_DIR, VITIS_HLS_PATH, n_jobs=24)
    # compute_fpga_base_benchmark(
    #     combos, RESULTS_DIR, BUILD_DIR, VITIS_HLS_PATH, n_jobs=24
    # )
    compute_fpga_par_benchmark(
        combos, RESULTS_DIR, BUILD_DIR, VITIS_HLS_PATH, n_jobs=24
    )
