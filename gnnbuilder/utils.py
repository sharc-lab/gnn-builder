import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import degree as pyg_degree


def compute_max_nodes_and_edges(dataset):
    max_node = 0
    max_edge = 0
    for data in dataset:
        max_node = max(max_node, data.num_nodes)
        max_edge = max(max_edge, data.num_edges)
    return max_node, max_edge


def compute_average_nodes_and_edges(dataset, round_val: bool = True):
    avg_nodes = 0.0
    avg_edges = 0.0
    count = 0
    for data in dataset:
        avg_nodes += data.num_nodes
        avg_edges += data.num_edges
        count += 1
    avg_nodes /= count
    avg_edges /= count
    if round_val:
        avg_nodes = int(round(avg_nodes))
        avg_edges = int(round(avg_edges))
    return avg_nodes, avg_edges


def compute_median_nodes_and_edges(dataset, round_val: bool = True):
    nodes = []
    edges = []
    for data in dataset:
        nodes.append(data.num_nodes)
        edges.append(data.num_edges)
    median_nodes = int(np.median(nodes))
    median_edges = int(np.median(edges))
    if round_val:
        median_nodes = int(round(median_nodes))
        median_edges = int(round(median_edges))
    return median_nodes, median_edges


def compute_degree(graph):
    in_degree = torch.zeros(graph.num_nodes)
    out_degree = torch.zeros(graph.num_nodes)
    for i in range(graph.num_edges):
        source = graph.edge_index[0, i]
        dest = graph.edge_index[1, i]
        in_degree[dest] += 1
        out_degree[source] += 1
    return in_degree.tolist(), out_degree.tolist()


def compute_average_degree(dataset, round_val=True):
    avg_degree = 0
    count = 0
    for data in dataset:
        in_degrees, _ = compute_degree(data)
        graph_avg_degree = np.mean(in_degrees)
        avg_degree += graph_avg_degree
        count += 1
    avg_degree /= count
    if round_val:
        avg_degree = int(np.ceil(avg_degree))
    return avg_degree


def compute_median_degree(dataset):
    degrees = []
    for data in dataset:
        in_degrees, _ = compute_degree(data)
        degrees.append(np.median(in_degrees))
    median_degree = np.median(degrees)
    median_degree_round = int(np.ceil(median_degree))
    return median_degree_round


def compute_in_deg_histogram(dataset):
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in dataset:
        d = pyg_degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = pyg_degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg


def layer_param_name_combiner(layer_name, param_name):
    return f"{layer_name}_{param_name.replace('.', '_')}"


def read_file(file_path):
    with open(file_path, "r") as f:
        return f.read()


def write_file(file_path, content):
    with open(file_path, "w") as f:
        f.write(content)


def serialize_tensor(param: torch.Tensor, fp: Path, np_type=np.float32):
    np_array: np.ndarray = param.detach().numpy().astype(np_type)
    np_array.tofile(fp)


def extract_data_from_csynth_report(top_report_path: Path):
    with open(top_report_path, "r") as f:
        xml_content_top = f.read()

    top_root = ET.fromstring(xml_content_top)

    load_parameters_fp = top_report_path.parent / "load_parameters_csynth.xml"
    with open(load_parameters_fp, "r") as f:
        xml_content_load_parameters = f.read()

    load_parameters_root = ET.fromstring(xml_content_load_parameters)

    clock_period = float(
        top_root.find("UserAssignments").find("TargetClockPeriod").text
    )
    worst_case_runtime_cycles_top = float(
        top_root.find("PerformanceEstimates")
        .find("SummaryOfOverallLatency")
        .find("Worst-caseLatency")
        .text
    )
    worst_case_runtime_cycles_load_parameters = float(
        load_parameters_root.find("PerformanceEstimates")
        .find("SummaryOfOverallLatency")
        .find("Worst-caseLatency")
        .text
    )
    worst_case_runtime_cycles = (
        worst_case_runtime_cycles_top - worst_case_runtime_cycles_load_parameters
    )
    worst_case_runtime_ns = worst_case_runtime_cycles * clock_period
    worst_case_runtime = worst_case_runtime_ns / 1e9

    bram = int(top_root.find("AreaEstimates").find("Resources").find("BRAM_18K").text)
    dsp = int(top_root.find("AreaEstimates").find("Resources").find("DSP").text)
    ff = int(top_root.find("AreaEstimates").find("Resources").find("FF").text)
    lut = int(top_root.find("AreaEstimates").find("Resources").find("LUT").text)
    uram = int(top_root.find("AreaEstimates").find("Resources").find("URAM").text)

    synth_data = {
        "clock_period": clock_period,
        "worst_case_runtime_cycles_top": worst_case_runtime_cycles_top,
        "worst_case_runtime_cycles_load_parameters": (
            worst_case_runtime_cycles_load_parameters
        ),
        "worst_case_runtime_cycles": worst_case_runtime_cycles,
        "worst_case_runtime_ns": worst_case_runtime_ns,
        "worst_case_runtime": worst_case_runtime,
        "bram": bram,
        "dsp": dsp,
        "ff": ff,
        "lut": lut,
        "uram": uram,
    }

    return synth_data
