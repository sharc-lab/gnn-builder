import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import seaborn as sns

import os
import glob
import itertools
from pprint import pp
from pathlib import Path
import re


def split(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])


RUNTIME_TYPES = ["pyg_cpu", "pyg_gpu", "cpp_cpu", "fpga_base", "fpga_dse"]


def read_runtime_results_file(fp: Path):
    with open(fp, "r") as f:
        average_runtime = float(f.readline().split()[-1])
        measurements = [float(line.split()[-1]) for line in f.readlines()]
    return average_runtime, measurements


def read_energy_results_file(fp: Path):
    with open(fp, "r") as f:
        average_power = float(f.readline().split()[-1])
        total_energy = float(f.readline().split()[-1])
        total_time = float(f.readline().split()[-1])

    return average_power, total_energy, total_time


def read_resource_results_file(fp: Path):
    with open(fp, "r") as f:
        lines = f.readlines()
    data_raw = [line.split() for line in lines]
    data = [(x[0], int(x[1])) for x in data_raw]
    pp(data)
    return data


def process_runtime_results(results_dir: Path, figures_dir: Path):
    runtime_data: list[dict] = []

    runtime_files_str = glob.glob(os.path.join(results_dir, "runtime_*.txt"))
    runtime_files = [Path(f) for f in runtime_files_str]

    for fp in runtime_files:
        fp_stem = fp.stem
        runtime_type_name = fp_stem.removeprefix("runtime_")
        runtime_type_name = split(runtime_type_name, "_", 2)[0]
        model_name = fp_stem.split("_")[3]
        dataset_name = fp_stem.split("_")[4]

        if runtime_type_name in ["pyg_cpu", "pyg_gpu"]:
            bs = fp_stem.split("_")[5].replace("b", "")
            if bs != "1":
                continue

        average_runtime, measurements = read_runtime_results_file(fp)
        runtime_data.append(
            {
                "runtime_type": runtime_type_name,
                "model": model_name,
                "dataset": dataset_name,
                "runtime": average_runtime * 1000,
            }
        )

    runtime_df = pd.DataFrame(runtime_data)
    # qm9_subset_df = runtime_df[runtime_df["dataset"] == "qm9"]
    runtime_df.to_csv("./figures/runtime_results.csv", index=False)
    print(runtime_df)

    # pivot table
    # rows are models
    # cols are runtime_type
    # values are runtime
    # values are aggregated by mean
    runtime_df_pivot = runtime_df.pivot_table(
        index="model", columns="runtime_type", values="runtime", aggfunc=np.mean
    )
    print(runtime_df_pivot)
    runtime_df_pivot.to_csv("./figures/runtime_results_pivot.csv")

    runtime_df_pivot["fpga_par_speedup_over_pyg_cpu"] = (
        runtime_df_pivot["pyg_cpu"] / runtime_df_pivot["fpga_par"]
    )
    runtime_df_pivot["fpga_par_speedup_over_pyg_gpu"] = (
        runtime_df_pivot["pyg_gpu"] / runtime_df_pivot["fpga_par"]
    )
    runtime_df_pivot["fpga_par_speedup_over_cpp_cpu"] = (
        runtime_df_pivot["cpp_cpu"] / runtime_df_pivot["fpga_par"]
    )

    runtime_seepup_df = runtime_df_pivot[
        [
            "fpga_par_speedup_over_pyg_cpu",
            "fpga_par_speedup_over_pyg_gpu",
            "fpga_par_speedup_over_cpp_cpu",
        ]
    ]
    # bottom row is geo mean of columns
    runtime_seepup_df.loc["geomean"] = runtime_seepup_df.apply(
        lambda x: np.exp(np.mean(np.log(x)))
    )
    runtime_seepup_df.to_csv("./figures/runtime_speedup_results.csv")
    # to latex
    # use \midrule to separate rows
    # use \cmidrule{1-3} to separate columns
    # use \multicolumn{3}{c}{\textbf{Runtime (ms)}} to add a header
    # last row has a \midrule before it
    runtime_seepup_df_latex = runtime_seepup_df.to_latex(
        float_format="{:0.2f}x".format,
        multicolumn_format="c",
        column_format="c | ccc",
        caption="FPGA-PAR Runtime speedup over PyG CPU, PyG GPU, and C++ CPU runtimes.",
        label="tab:runtime_speedup",
    )
    with open("./figures/runtime_speedup_results.tex", "w") as f:
        f.write(runtime_seepup_df_latex)
    print(runtime_seepup_df)

    font = {"size": 16}
    matplotlib.rc("font", **font)

    plt.style.use("seaborn-deep")
    colors = ["#90e0ef", "#00b4d8", "#95d5b2", "#e5383b", "#f48c06"]
    title_map = {
        "qm9": "QM9",
        "esol": "ESOL",
        "freesolv": "FreeSolv",
        "lipo": "Lipophilicity",
        "hiv": "HIV",
    }
    legend_map = {
        "pyg_cpu": "PYG-CPU",
        "pyg_gpu": "PYG-GPU",
        "cpp_cpu": "CPP-CPU",
        "fpga_base": "FPGA-Base",
        "fpga_dse": "FPGA-DSE",
        "fpga_par": "FPGA-Parallel",
    }

    g = sns.catplot(
        row="dataset",
        x="model",
        y="runtime",
        hue="runtime_type",
        kind="bar",
        data=runtime_df,
        legend=False,
        sharex=False,
        sharey=True,
        palette=colors,
    )

    fig = g.figure
    fig.set_size_inches(12, 12)

    axes = g.axes.ravel()
    axes_dict = g.axes_dict

    for title, ax in axes_dict.items():
        ax.set_ylabel("Runtime (ms)")
        ax.set_ylim(1e-1, 50 * (10**1))
        ax.set_yscale("log", subs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.set_yticks([1e-1, 1e0, 1e1, 1e2])
        ax.set_yticklabels(["0.1", "1", "10", "100"])

        # ax.set_ylim(1e-2, 1*(10**1))

        x_labels = [x.get_text().upper() for x in ax.get_xticklabels()]
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("")

        ax.set_axisbelow(True)
        ax.grid(axis="y")

        title_str = title_map[title]
        title_str = f"Dataset: {title_str}"

        ax.set_title(title_str, y=0.95)

        for container in ax.containers:
            labels = ax.bar_label(container, fmt="%.2f", fontsize=12, padding=2)
            for label in labels:
                label.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.1))

    fig_legend = axes[-1].legend(
        loc="center",
        bbox_to_anchor=(0.5, -0.5),
        shadow=False,
        ncol=5,
    )
    # make the legend text all uppercase
    for text_obj in fig_legend.get_texts():
        new_str = legend_map[text_obj.get_text()]
        text_obj.set_text(new_str)

    # axs.set_ylim(0, 10)
    # axs.set_ylabel("Runtime (ms)")
    # axs.set_xlabel("Model")
    # axs.set_title("Runtime Comparison")
    # axs.legend(loc="upper left")

    fig.suptitle("Model Runtime Comparison", y=0.98, fontsize=24)

    plt.tight_layout(h_pad=0.8, w_pad=0)
    plt.savefig(figures_dir / "runtime_barplot.png", dpi=300)


def process_resource_results(results_dir: Path, figures_dir: Path):
    resource_available_u50 = {
        "bram": 2688,
        "dsp": 5952,
        "ff": 1743360,
        "lut": 871680,
        "uram": 640,
    }

    resources_available_u280 = {
        "bram": 4032,
        "dsp": 9024,
        "ff": 2607360,
        "lut": 1303680,
        "uram": 960,
    }

    resource_available = resources_available_u280

    legend_map = {
        "pyg_cpu": "PYG-CPU",
        "pyg_gpu": "PYG-GPU",
        "cpp_cpu": "CPP-CPU",
        "fpga_base": "FPGA-Base",
        "fpga_dse": "FPGA-DSE",
        "fpga_par": "FPGA-Parallel",
    }

    resource_data: list[dict] = []

    resource_files_str = glob.glob(os.path.join(results_dir, "resources_*.txt"))
    resource_files = [Path(f) for f in resource_files_str]
    # pp(resource_files)

    for fp in resource_files:
        fp_stem = fp.stem
        runtime_type_name = fp_stem.removeprefix("resources_")
        runtime_type_name = split(runtime_type_name, "_", 2)[0]
        model_name = fp_stem.split("_")[3]
        dataset_name = fp_stem.split("_")[4]

        if runtime_type_name in ["pyg_cpu", "pyg_gpu"]:
            bs = fp_stem.split("_")[5].replace("b", "")
            if bs != "1":
                continue

        resources = read_resource_results_file(fp)
        for r in resources:
            resource_data.append(
                {
                    "runtime_type": runtime_type_name,
                    "model": model_name,
                    "dataset": dataset_name,
                    "resource": r[0],
                    "value": r[1] / resource_available[r[0]],
                }
            )

    resource_df = pd.DataFrame(resource_data)
    resource_df = resource_df[resource_df["resource"] != "uram"]
    print(
        resource_df.pivot_table(
            index=["runtime_type", "model", "dataset"],
            values="value",
            columns="resource",
        )
    )
    print(resource_df)

    # index is (runtime_type, model)
    # columns are the resources
    # values are the resource utilization
    # data is average utilization across all datasets
    # resource_df_pivot = resource_df.pivot_table(index=["runtime_type", "model"], values="value", columns="resource")
    # resource_df_pivot = resource_df_pivot.sort_values(by=["runtime_type", "model"])

    # # rename resource names
    # resource_df_pivot = resource_df_pivot.rename(columns={
    #     "bram": "BRAM",
    #     "dsp": "DSP",
    #     "ff": "FF",
    #     "lut": "LUT"
    # })

    # # rename runtime type names
    # resource_df_pivot = resource_df_pivot.rename(index={
    #     "fpga_base": "FPGA-Base",
    #     "fpga_par": "FPGA-Parallel",
    # })

    # # rename model names
    # resource_df_pivot = resource_df_pivot.rename(index={
    #     "gcn": "GCN",
    #     "gin": "GIN",
    #     "pna": "PNA",
    #     "sage": "SAGE"
    # })
    resource_df_pivot = resource_df.pivot_table(
        index=["model"], values="value", columns=["runtime_type", "resource"]
    )
    resource_df_pivot = resource_df_pivot.sort_values(by=["model"])
    resource_df_pivot = resource_df_pivot.rename(
        index={"gcn": "GCN", "gin": "GIN", "pna": "PNA", "sage": "SAGE"}
    )
    resource_df_pivot = resource_df_pivot.rename(
        columns={"bram": "BRAM", "dsp": "DSP", "ff": "FF", "lut": "LUT"}
    )
    resource_df_pivot = resource_df_pivot.rename(
        columns={
            "fpga_base": "FPGA-Base",
            "fpga_par": "FPGA-Parallel",
        }
    )
    print(resource_df_pivot)

    print(resource_df_pivot)
    # write to csv
    resource_df_pivot.to_csv(figures_dir / "resource_utilization_pivot.csv")
    # to latex
    # values are decinamsl that need to be multiplied by 100 and rounded to 0 decimal places
    resource_df_pivot_latex = resource_df_pivot.to_latex(
        float_format=lambda x: f"{round(x * 100, 1)}%",
        multicolumn_format="c|",
        column_format="l|cccc|cccc|",
        multirow=True,
        caption="Resource usage of FPGA-Base and FPGA-Parallel model implementations.",
        label="tab:resources",
    )
    print(resource_df_pivot_latex)
    with open(figures_dir / "resource_utilization_pivot.tex", "w") as f:
        f.write(resource_df_pivot_latex)

    font = {"size": 14}
    matplotlib.rc("font", **font)

    colors = ["#b5e48c", "#76c893", "#34a0a4", "#1a759f"]

    g = sns.catplot(
        col="resource",
        col_wrap=2,
        x="model",
        y="value",
        hue="runtime_type",
        kind="bar",
        data=resource_df,
        sharex=False,
        sharey=False,
        legend=False,
        palette=colors,
        errorbar=None,
    )

    fig = g.figure
    fig.set_size_inches(8, 6)

    axes = g.axes.ravel()
    axes_dict = g.axes_dict
    for title, ax in axes_dict.items():
        resource_lim = resource_available[title]
        # plot horizontal line at resource limit
        ax.axhline(y=1.0, color="#e5383b", linestyle="--")

        ax.set_title(title.upper(), y=1.0)

        ax.set_ylabel("")
        ax.set_xlabel("")

        # if title in ["lut", "ff"]:
        #     ylabels = ['{:,.2f}'.format(x) + 'K' for x in ax.get_yticks()/1000]
        #     ax.set_yticklabels(ylabels)

        ax.set_ylim(0, 1.2)
        # y axis format as percentage
        y_labels = ["{:,.0f}".format(x * 100) + "%" for x in ax.get_yticks()]
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        ax.set_yticklabels(y_labels)

        for container in ax.containers:
            labels = ax.bar_label(container, fontsize=9)
            for l in labels:
                if l.get_text() == "":
                    continue
                num = float(l.get_text()) * 100
                l.set_text(f"{num:.1f}%")
                # l.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.01))

        ax.set_axisbelow(True)
        ax.grid(axis="y")

        ax.set_xticklabels([x.get_text().upper() for x in ax.get_xticklabels()])

    # fig_legend = axes[-1].legend(
    #         loc="center",
    #         bbox_to_anchor=(0.5, -0.5),
    #         shadow=False,
    #         ncol=5,
    #     )
    l = fig.legend(
        *axes[-1].get_legend_handles_labels(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.95),
        fontsize=8,
    )

    # make the legend text all uppercase
    for text_obj in l.get_texts():
        new_str = legend_map[text_obj.get_text()]
        text_obj.set_text(new_str)

    fig.suptitle("FPGA Model Resource Usage", y=0.99, fontsize=16)

    plt.tight_layout(h_pad=0.7, w_pad=0.5)
    plt.savefig(figures_dir / "resource_barplot.png", dpi=300)


def process_batch_results(
    results_dir: Path, results_batch_dir: Path, figures_dir: Path
):
    runtime_data: list[dict] = []

    runtime_files_str = glob.glob(os.path.join(results_dir, "runtime_*.txt"))
    runtime_files = [Path(f) for f in runtime_files_str]

    for fp in runtime_files:
        fp_stem = fp.stem
        runtime_type_name = fp_stem.removeprefix("runtime_")
        runtime_type_name = split(runtime_type_name, "_", 2)[0]
        model_name = fp_stem.split("_")[3]
        dataset_name = fp_stem.split("_")[4]

        average_runtime, measurements = read_runtime_results_file(fp)
        runtime_data.append(
            {
                "runtime_type": runtime_type_name,
                "model": model_name,
                "dataset": dataset_name,
                "runtime": average_runtime * 1000,
                "bs": 1,
            }
        )

    runtime_files_batch_str = glob.glob(
        os.path.join(results_batch_dir, "runtime_*.txt")
    )
    runtime_files_batch = [Path(f) for f in runtime_files_batch_str]
    for fp in runtime_files_batch:
        fp_stem = fp.stem
        runtime_type_name = fp_stem.removeprefix("runtime_")
        runtime_type_name = split(runtime_type_name, "_", 2)[0]
        model_name = fp_stem.split("_")[3]
        dataset_name = fp_stem.split("_")[4]

        average_runtime, measurements = read_runtime_results_file(fp)
        runtime_data.append(
            {
                "runtime_type": runtime_type_name,
                "model": model_name,
                "dataset": dataset_name,
                "runtime": average_runtime * 1000,
                "bs": 4,
            }
        )

    runtime_df = pd.DataFrame(runtime_data)
    # avaege runtime over all datasets
    runtime_df = (
        runtime_df.groupby(["model", "runtime_type", "bs"]).mean().reset_index()
    )
    print(runtime_df)

    runtime_df_fpga = runtime_df[
        runtime_df["runtime_type"].isin(["fpga_base", "fpga_par"])
    ]
    runtime_df_bs_4 = runtime_df[runtime_df["bs"] == 4]

    runtime_bs_df = pd.concat([runtime_df_fpga, runtime_df_bs_4])
    # merge the runtime_type and bs columns
    runtime_bs_df["runtime_type"] = (
        runtime_bs_df["runtime_type"] + "_" + runtime_bs_df["bs"].astype(str)
    )
    runtime_bs_df = runtime_bs_df.drop(columns=["bs"])

    legend_map_batch = {
        "pyg_cpu_4": "PyG-CPU\nbs=4",
        "pyg_gpu_4": "PyG-GPU\nbs=4",
        "fpga_base_1": "FPGA-Base",
        "fpga_par_1": "FPGA-Parallel",
    }

    model_map = {
        "gcn": "GCN",
        "gin": "GIN",
        "pna": "PNA",
        "sage": "GraphSAGE",
    }

    colors_new = ["#90e0ef", "#00b4d8", "#e5383b", "#f48c06"]

    g = sns.catplot(
        row="model",
        x="runtime_type",
        y="runtime",
        kind="bar",
        data=runtime_bs_df,
        sharex=False,
        sharey=False,
        legend=False,
        order=["pyg_cpu_4", "pyg_gpu_4", "fpga_base_1", "fpga_par_1"],
        palette=colors_new,
    )

    fig = g.figure
    fig.set_size_inches(6, 6)

    axes = g.axes.ravel()
    axes_dict = g.axes_dict
    for title, ax in axes_dict.items():
        ax.set_ylabel("Runtime (ms)")
        ax.set_ylim(1e-1, 20 * (10**2))
        ax.set_yscale("log", subs=[3, 4, 5, 6, 7, 8, 9])
        ax.set_yticks(
            [
                1e-1,
                1e0,
                1e1,
                1e2,
                1e3,
            ]
        )
        ax.set_yticklabels(["0.1", "1", "10", "100", "1000"])

        ax.set_xticklabels(
            [legend_map_batch[x.get_text()] for x in ax.get_xticklabels()]
        )
        ax.set_title(model_map[title])

        for container in ax.containers:
            labels = ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
            for label in labels:
                label.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.1))

        ax.set_axisbelow(True)
        ax.grid(axis="y")

    plt.tight_layout()
    plt.savefig(figures_dir / "runtime_barplot_with_batch.png", dpi=300)


def process_energy_results(results_dir, results_batch_dir, figures_dir):
    energy_data: list[dict] = []

    energy_files_batch_str = glob.glob(os.path.join(results_batch_dir, "energy_*.txt"))
    energy_files_batch = [Path(f) for f in energy_files_batch_str]
    for fp in energy_files_batch:
        fp_stem = fp.stem
        runtime_type_name = fp_stem.removeprefix("energy_")
        runtime_type_name = split(runtime_type_name, "_", 2)[0]
        model_name = fp_stem.split("_")[3]
        dataset_name = fp_stem.split("_")[4]
        bs = fp_stem.split("_")[5].replace("b", "")

        if runtime_type_name in ["pyg_cpu", "pyg_gpu"]:
            bs = fp_stem.split("_")[5].replace("b", "")
            if bs != "1":
                continue

        average_power, total_energy, total_time = read_energy_results_file(fp)

        energy_data.append(
            {
                "runtime_type": runtime_type_name,
                "model": model_name,
                "dataset": dataset_name,
                "bs": bs,
                "average_power": average_power,
            }
        )

    energy_df = pd.DataFrame(energy_data)
    # only keep rows for pyg_gpu
    energy_df = energy_df[energy_df["runtime_type"] == "pyg_gpu"]
    # avaege power over all datasets for each model and batch size
    # energy_df = energy_df.groupby(["model", "runtime_type", "bs"]).mean().reset_index()
    energy_df = energy_df.groupby(["model", "runtime_type"]).mean().reset_index()

    print(energy_df)

    # energy_df_pivot
    energy_df_pivot = energy_df.pivot(
        index="model", columns="runtime_type", values="average_power"
    )
    print(energy_df_pivot)
    print(energy_df_pivot.to_latex(float_format="%.2f", escape=False))

    colors_new = ["#90e0ef", "#00b4d8", "#e5383b", "#f48c06"]
    # colors_new = ["#90e0ef", "#00b4d8"]

    model_map = {
        "gcn": "GCN",
        "gin": "GIN",
        "pna": "PNA",
        "sage": "GraphSAGE",
    }

    legend_map = {
        "pyg_cpu": "PyG-CPU",
        "pyg_gpu": "PyG-GPU",
        "fpga_base": "FPGA-Base",
        "fpga_par": "FPGA-Parallel",
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.barplot(
        data=energy_df,
        x="model",
        y="average_power",
        hue="runtime_type",
        palette=colors_new,
        ax=ax,
    )

    ax.set_ylabel("Average Power (W)")
    ax.set_xlabel("Model Architecture")

    ax.set_xticklabels([model_map[x.get_text()] for x in ax.get_xticklabels()])

    ax.set_ylim(0, 80)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])

    # for each bar, add the value
    for container in ax.containers:
        labels = ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
        for label in labels:
            label.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.1))

    ax.set_axisbelow(True)
    ax.grid(axis="y")

    h, l = ax.get_legend_handles_labels()
    l_new = [legend_map[x] for x in l]

    ax.legend(
        h,
        l_new,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        # frameon=False,
    )
    ax.set_title("Power Consumption of Different Model Implementations")

    plt.tight_layout()
    plt.savefig(figures_dir / "energy_barplot.png", dpi=300)


if __name__ == "__main__":
    RESULTS_DIR = Path("./results")
    if not RESULTS_DIR.exists():
        raise Exception(f"{RESULTS_DIR} does not exist")

    RESULTS_BATCH_DIR = Path("./results_batch")
    if not RESULTS_BATCH_DIR.exists():
        raise Exception(f"{RESULTS_BATCH_DIR} does not exist")

    FIGURES_DIR = Path("./figures")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # process_runtime_results(RESULTS_DIR, FIGURES_DIR)
    # process_resource_results(RESULTS_DIR, FIGURES_DIR)
    # process_batch_results(RESULTS_DIR, RESULTS_BATCH_DIR, FIGURES_DIR)
    process_energy_results(RESULTS_DIR, RESULTS_BATCH_DIR, FIGURES_DIR)
