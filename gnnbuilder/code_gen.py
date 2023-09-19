import os
import shutil
import subprocess
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

import jinja2
import numpy as np
import torch
from torch_geometric.data import Dataset, InMemoryDataset

from .models import GNNModel
from .utils import (
    extract_data_from_csynth_report,
    layer_param_name_combiner,
    serialize_tensor,
    write_file,
)

CURRENT_DIR = Path(__file__).parent
GNN_BUILDER_LIB = CURRENT_DIR / "gnn_builder_lib" / "gnn_builder_lib.h"

template_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(searchpath=CURRENT_DIR / "templates"),
    trim_blocks=True,
    lstrip_blocks=True,
)

MODEL_H_TEMPLATE = template_env.get_template("model.h.jinja")
MODEL_CC_TEMPLATE = template_env.get_template("model.cpp.jinja")
MODEL_TB_TEMPLATE = template_env.get_template("model_tb.cpp.jinja")
MAKEFILE_TEMPLATE = template_env.get_template("makefile_testbench.jinja")
RUN_HLS_TEMPLATE = template_env.get_template("run_hls.tcl.jinja")
MAKEFILE_VITIS_TEMPLATE = template_env.get_template("makefile_vitis.jinja")


class FPX:
    def __init__(self, W: int = 32, I: int = 16, Q: str = "AP_TRN", O: str = "AP_WRAP"):
        self.W = W
        self.I = I
        self.Q = Q
        self.O = O

        if I > 33:
            raise Exception("I must be <= 33")
        if W - I > 32:
            raise Exception("W-I must be <= 32")

    def __str__(self):
        return f"ap_fixed<{self.W},{self.I},{self.Q},{self.O}>"


SUPPORTED_FPGA_PARTS = [
    "xcu50-fsvh2104-2-e",
    "xcu280-fsvh2892-2L-e",
    # "xilinx_u280_gen3x16_xdma_1_202211_1"
]


class Project:
    def __init__(
        self,
        name: str,
        model: GNNModel,
        pyg_output_encoding: str,
        vitis_hls_path: Path,
        build_dir: Path,
        dataset: Optional[Union[Dataset, InMemoryDataset]] = None,
        max_nodes: int = 500,
        max_edges: int = 500,
        num_nodes_guess: Optional[int] = None,
        num_edges_guess: Optional[int] = None,
        degree_guess: Optional[int] = None,
        float_or_fixed: str = "float",
        fpx: FPX = FPX(W=32, I=16),
        clock_speed: float = 3.33,
        fpga_part: str = "xcu50-fsvh2104-2-e",
        n_jobs: int = 1,
    ):
        self.model = model
        self.dataset = dataset
        self.name = name
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        self.num_nodes_guess = num_nodes_guess
        self.num_edges_guess = num_edges_guess
        if self.num_nodes_guess is None:
            self.num_nodes_guess = self.max_nodes
        if self.num_edges_guess is None:
            self.num_edges_guess = self.max_edges

        self.degree_guess = degree_guess
        if self.degree_guess is None:
            self.degree_guess = self.max_nodes

        self.pyg_output_encoding = pyg_output_encoding
        valid_output_encodings = [
            "regression",
            "classification_integer",
            "classification_onehot",
        ]
        if self.pyg_output_encoding not in valid_output_encodings:
            raise ValueError(
                f"pyg_output_encoding must be one of {valid_output_encodings}"
            )

        self.vitis_hls_path = vitis_hls_path
        self.build_dir = build_dir

        self.float_or_fixed = float_or_fixed
        self.fpx = fpx

        vlaid_percision = ["float", "fixed"]
        if float_or_fixed not in vlaid_percision:
            raise ValueError(f"float_or_fixed must be one of {vlaid_percision}")

        self.clock_speed = clock_speed
        if self.clock_speed <= 0:
            raise ValueError("clock_speed must be > 0")

        self.fpga_part = fpga_part
        if self.fpga_part not in SUPPORTED_FPGA_PARTS:
            raise ValueError(f"fpga_part must be one of {SUPPORTED_FPGA_PARTS}")

        self.n_jobs = n_jobs
        if self.n_jobs <= 0:
            raise ValueError("n_jobs must be > 0")

    def validate_project(self):
        if self.name is None:
            raise Exception("No name is set.")
        if self.dataset is None:
            raise Exception("No dataset is set.")
        if self.model is None:
            raise Exception("No model is set.")

    @cached_property
    def model_dir(self):
        return self.build_dir / self.name

    @cached_property
    def template_dict(self):
        template_dict = {}

        template_dict["vitis_hls_path"] = self.vitis_hls_path
        template_dict["build_dir"] = str(self.build_dir)
        template_dict["model_dir"] = str(self.model_dir)

        template_dict["model_top_name"] = self.name
        template_dict["max_nodes"] = self.max_nodes
        template_dict["max_edges"] = self.max_edges
        template_dict["num_nodes_guess"] = self.num_nodes_guess
        template_dict["num_edges_guess"] = self.num_edges_guess
        template_dict["degree_guess"] = self.degree_guess

        template_dict["input_node_features_dim"] = self.model.input_node_features_dim
        template_dict["output_features_dim"] = self.model.output_features_dim

        # pp(self.model.layers)
        # pp(self.model.layer_names)
        # pp(self.model.layer_parameter_names_flat)

        # pp(self.model.layer_parameter_shapes)
        # pp(self.model.layer_parameter_shapes_flat)

        model_parameters_info: list[dict] = []
        for i, parameter_name in enumerate(self.model.layer_parameter_names_flat):
            model_parameters_info.append(
                {
                    "name": parameter_name,
                    "shape": self.model.layer_parameter_shapes_flat[i],
                    "shape_len": len(self.model.layer_parameter_shapes_flat[i]),
                    "size": np.prod(self.model.layer_parameter_shapes_flat[i]),
                }
            )
        template_dict["model_parameters"] = model_parameters_info

        node_emb_buffer_size = self.model.gnn_hidden_dim
        template_dict["node_emb_buffer_size"] = node_emb_buffer_size

        template_dict["model"] = self.model

        template_dict["float_or_fixed"] = self.float_or_fixed
        template_dict["fpx"] = self.fpx

        template_dict["clock_speed"] = self.clock_speed
        template_dict["fpga_part"] = self.fpga_part

        template_dict["n_jobs"] = self.n_jobs

        return template_dict

    def gen_hw_model(self):
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        shutil.copy(GNN_BUILDER_LIB, self.model_dir)

        template_dict = self.template_dict

        model_h_template_render = MODEL_H_TEMPLATE.render(template_dict)
        write_file(self.model_dir / "model.h", model_h_template_render)

        model_cc_template_render = MODEL_CC_TEMPLATE.render(template_dict)
        write_file(self.model_dir / "model.cpp", model_cc_template_render)

    def gen_testbench(self, gen_testbench_data=True):
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        template_dict = self.template_dict

        model_tb_template_render = MODEL_TB_TEMPLATE.render(template_dict)
        write_file(self.model_dir / "model_tb.cpp", model_tb_template_render)

        if gen_testbench_data:
            self.gen_testbench_data()

    def gen_testbench_data(self):
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        tb_data_dir = self.model_dir / "tb_data"
        os.makedirs(tb_data_dir, exist_ok=True)

        tb_data_model_parameters_dir = tb_data_dir / "model_parameters"
        os.makedirs(tb_data_model_parameters_dir, exist_ok=True)

        # layer_parameters
        # pp(self.model.layers)
        for layer in self.model.layers:
            for param in self.model.layer_parameters[layer]:
                layer_param_name = layer_param_name_combiner(layer, param[0])
                param_data_path = (
                    tb_data_model_parameters_dir / f"{layer_param_name}.bin"
                )
                param_tensor = param[1]
                serialize_tensor(param_tensor, param_data_path)

        # graph_data
        tb_data_graphs_dir = tb_data_dir / "graphs"
        os.makedirs(tb_data_graphs_dir, exist_ok=True)

        dataset_info_fp = tb_data_dir / "dataset_info.txt"
        dataset_indices = list(self.dataset.indices())
        with open(dataset_info_fp, "w") as f:
            f.write(f"num_graphs {len(dataset_indices)}\n")
            for index in dataset_indices:
                f.write(f"{index}\n")

        for i, idx in enumerate(dataset_indices):
            graph_idx_str = str(idx)
            graph = self.dataset[idx]
            graph_coo = graph.edge_index.T.type(torch.int32)
            graph_num_nodes = graph.num_nodes
            graph_num_edges = graph.num_edges
            graph_info = torch.tensor([graph_num_nodes, graph_num_edges]).type(
                torch.int32
            )
            graph_node_features = graph.x
            task_golden_output: Optional[torch.Tensor] = None
            if self.pyg_output_encoding == "regression":
                task_golden_output = graph.y.type(torch.float32).view(-1)
            elif self.pyg_output_encoding == "classification_integer":
                num_classes = self.dataset.num_classes
                task_golden_output = torch.zeros(num_classes).type(torch.float32)
                task_golden_output[graph.y.type(torch.long)[0]] = 1.0
            elif self.pyg_output_encoding == "classification_onehot":
                task_golden_output = graph.y.type(torch.float32).view(-1)
                assert task_golden_output.shape[0] == self.dataset.num_classes
            graph_model_golden_output = (
                self.model(
                    graph.x.type(torch.float32), graph.edge_index.type(torch.int64)
                )
                .detach()
                .view(-1)
            )

            graph_info_fp = tb_data_graphs_dir / f"graph_{graph_idx_str}_info.bin"
            graph_coo_fp = tb_data_graphs_dir / f"graph_{graph_idx_str}_coo.bin"
            graph_node_features_fp = (
                tb_data_graphs_dir / f"graph_{graph_idx_str}_node_features.bin"
            )
            graph_task_golden_output_fp = (
                tb_data_graphs_dir / f"graph_{graph_idx_str}_task_golden_output.bin"
            )
            graph_model_golden_output_fp = (
                tb_data_graphs_dir / f"graph_{graph_idx_str}_model_golden_output.bin"
            )

            # print(graph_info)
            serialize_tensor(graph_info, graph_info_fp, np_type=np.int32)
            serialize_tensor(graph_coo, graph_coo_fp, np_type=np.int32)
            serialize_tensor(graph_node_features, graph_node_features_fp)
            if task_golden_output is not None:
                serialize_tensor(task_golden_output, graph_task_golden_output_fp)
            serialize_tensor(graph_model_golden_output, graph_model_golden_output_fp)

    def gen_makefile(self):
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        template_dict = self.template_dict

        makefile_template_render = MAKEFILE_TEMPLATE.render(template_dict)
        write_file(self.model_dir / "makefile_testbench", makefile_template_render)

    def gen_vitis_hls_tcl_script(self):
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        template_dict = self.template_dict

        vitis_hls_tcl_script_template_render = RUN_HLS_TEMPLATE.render(template_dict)
        write_file(self.model_dir / "run_hls.tcl", vitis_hls_tcl_script_template_render)

    def build_and_run_testbench(self):
        files_to_check = [
            self.model_dir / "model_tb.cpp",
            self.model_dir / "model.h",
            self.model_dir / "model.cpp",
            self.model_dir / "makefile_testbench",
        ]

        for fp in files_to_check:
            if not fp.exists():
                raise Exception(
                    f"{self.name} - {fp} does not exist. Make sure you call the"
                    " gen_<...> functions to generate the model and testbench source"
                    " code."
                )

        proc_build = subprocess.run(
            [
                "make",
                "-f",
                "makefile_testbench",
                "run",
            ],
            cwd=self.model_dir,
            capture_output=True,
        )

        if proc_build.returncode != 0:
            print(f"return code: {proc_build.returncode}")
            print(proc_build.stdout.decode("utf-8"))
            print(proc_build.stderr.decode("utf-8"))
            raise Exception(f"{self.name} - Testbench build failed.")
        else:
            print(proc_build.stdout.decode("utf-8"))

        proc_run = subprocess.run(["./result"], cwd=self.model_dir, capture_output=True)
        if proc_run.returncode != 0:
            print(f"return code: {proc_run.returncode}")
            print(proc_run.stdout.decode("utf-8"))
            print(proc_run.stderr.decode("utf-8"))
            raise Exception(f"{self.name} - Testbench execution failed.")
        else:
            print(proc_run.stdout.decode("utf-8"))

        # read the output files
        model_output_mae_fp = self.model_dir / "tb_data" / "model_output_mae.txt"
        model_runtime_fp = self.model_dir / "tb_data" / "model_runtime.txt"

        model_output_mae = float(model_output_mae_fp.read_text().strip().split()[1])
        model_runtime = float(model_runtime_fp.read_text().strip().split()[1])

        data = {
            "model_output_mae": model_output_mae,
            "model_runtime": model_runtime,
        }

        return data

    def run_vitis_hls_synthesis(self, verbose=False):
        files_to_check = [
            self.model_dir / "gnn_builder_lib.h",
            self.model_dir / "model.h",
            self.model_dir / "model.cpp",
        ]

        for fp in files_to_check:
            if not fp.exists():
                raise Exception(
                    f"{self.name} - {fp} does not exist. Make sure you call the"
                    " gen_<...> functions to generate the model and testbench source"
                    " code."
                )

        proj_tcl_file = str((self.model_dir / "run_hls.tcl").resolve())

        print("Launching HLS synthesis...")
        proc = subprocess.run(
            ["vitis_hls", proj_tcl_file],
            cwd=self.model_dir,
            capture_output=True,
        )

        if proc.returncode != 0:
            print(f"return code: {proc.returncode}")
            if verbose:
                print(proc.stdout.decode("utf-8"))
                print(proc.stderr.decode("utf-8"))
            raise Exception(f"{self.name} - Vitis HLS synthesis failed.")
        else:
            if verbose:
                print(proc.stdout.decode("utf-8"))

        synth_report_fp = (
            self.model_dir
            / f"{self.name}_vitis_hls_project"
            / "solution1"
            / "syn"
            / "report"
            / f"{self.name}_top_csynth.xml"
        )
        if not synth_report_fp.exists():
            raise Exception(
                f"{self.name} - Can't find synthesis report file: {synth_report_fp}"
            )

        synth_data = extract_data_from_csynth_report(synth_report_fp)

        return synth_data

    def gen_makefile_vitis(self):
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        template_dict = self.template_dict

        makefile_template_render = MAKEFILE_VITIS_TEMPLATE.render(template_dict)
        write_file(self.model_dir / "makefile_vitis", makefile_template_render)

    def build_hw_kernel(self):
        files_to_check = [
            self.model_dir / "model.h",
            self.model_dir / "model.cpp",
            self.model_dir / "makefile_vitis",
        ]

        for fp in files_to_check:
            if not fp.exists():
                raise Exception(
                    f"{self.name} - {fp} does not exist. Make sure you call the"
                    " gen_<...> functions to generate the needed files."
                )

        # call the makefile_vitis makefile specificly
        proc_build = subprocess.run(
            [
                "make",
                "-f",
                "makefile_vitis",
                "all",
            ],
            cwd=self.model_dir,
            capture_output=True,
        )

        if proc_build.returncode != 0:
            print(f"return code: {proc_build.returncode}")
            print(proc_build.stdout.decode("utf-8"))
            print(proc_build.stderr.decode("utf-8"))
            raise Exception(f"{self.name} - Kernel build failed.")
        else:
            print(proc_build.stdout.decode("utf-8"))
