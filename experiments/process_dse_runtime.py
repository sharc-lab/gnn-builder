import os

# from pprint import pp
import pickle
import re
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from matplotlib import ticker, transforms

from experiments.process_dse_models import transform_x

DSE_MODEL_DIR = Path("./dse_models/")
with open(DSE_MODEL_DIR / "model_direct.pk", "rb") as f:
    model_latency_direct = pickle.load(f)

with open(DSE_MODEL_DIR / "model_bram.pk", "rb") as f:
    model_bram_direct = pickle.load(f)

model_transform_x = transform_x


def estimate_vitis_hls_runtime(n_designs, avg_single_runtime, cores=1):
    return (n_designs * avg_single_runtime) / cores


def extract_runtime_from_vitis_hls_log(log_file: Path):
    # print(log_file)
    # grep the log file for the runtime
    # "Total elapsed time: 261.83 seconds"
    pattern = r"Total elapsed time: (\d+(?:[.,]\d+)?) seconds"
    number_str = None
    with open(log_file, "r") as f:
        for line in f.readlines():
            if "Total elapsed time" in line:
                # print(line)
                number_str = re.search(pattern, line).group(1)
                break

    if number_str is None:
        raise Exception("Could not find runtime in log file")

    number_float = float(number_str)
    return number_float


def extract_vitis_hls_log_rutimes_from_build_dir(BUILD_DIR: Path):
    log_files = list(BUILD_DIR.rglob("**/vitis_hls.log"))
    # print(log_files)
    idxs = []
    for log_file in log_files:
        idx = log_file.parent.name.split("_")[-2]
        idxs.append(idx)
    # print(idxs)
    runtimes = [extract_runtime_from_vitis_hls_log(fp) for fp in log_files]
    return runtimes, idxs


def time_model_calls(model, x, trials=100):
    start = time.perf_counter_ns()
    for trials in range(trials):
        model.predict(x)
    end = time.perf_counter_ns()
    t_ns = (end - start) / trials
    t = t_ns / 1e9
    return t


def compute_config_model_prediction_data_and_time(row, trials=10):
    # pd.set_option('display.max_columns', None)

    x_cols = [
        "gnn_in_dim",
        "gnn_hidden_dim",
        "gnn_out_dim",
        "gnn_num_layers",
        "mlp_in_dim",
        "mlp_hidden_dim",
        "mlp_out_dim",
        "mlp_num_layers",
        "gnn_p_in",
        "gnn_p_hidden",
        "gnn_p_out",
        "mlp_p_in",
        "mlp_p_hidden",
        "mlp_p_out",
        "conv",
        "gnn_skip_connections",
        "median_nodes",
        "median_edges",
        "median_degree",
    ]

    x = pd.DataFrame([row], columns=x_cols)
    x = transform_x(x)
    # merge all rows into one use the first value in the colum that is not nan
    # s.loc[~s.isnull()].iloc[0]
    for col in x.columns:
        x[col] = x[col].fillna(x[col].loc[~x[col].isnull()].iloc[0])
    x = x.iloc[:1]
    # x = x.iloc[:1].to_numpy().reshape(1,-1)

    latency_pred = model_latency_direct.predict(x).item()
    bram_pred = model_bram_direct.predict(x).item()
    latency_time = time_model_calls(model_latency_direct, x, trials=trials)
    bram_time = time_model_calls(model_bram_direct, x, trials=trials)

    return latency_pred, bram_pred, latency_time, bram_time


if __name__ == "__main__":
    DSE_DATA_DIR = Path("./results_perf/")
    os.makedirs(DSE_DATA_DIR, exist_ok=True)
    dse_csv_fp = DSE_DATA_DIR / "dse_data.csv"

    if not dse_csv_fp.exists():
        PERF_DATA_BUILD_DIR = Path("/usr/scratch/skaram7/gnnb_perf_builds/")
        runtimes, idxs = extract_vitis_hls_log_rutimes_from_build_dir(
            PERF_DATA_BUILD_DIR
        )
        vitis_tool_runtime_df = pd.DataFrame(
            {"build_idx": idxs, "vitis_tool_runtime": runtimes}
        )
        vitis_tool_runtime_df = vitis_tool_runtime_df.astype({"build_idx": int})

        print(model_latency_direct)
        print(model_bram_direct)
        print(model_transform_x)

        PERF_DATA_DIR = Path("./results_perf/")

        data = pd.read_csv(PERF_DATA_DIR / "perf_data.csv")
        data = pd.merge(
            data, vitis_tool_runtime_df, how="left", left_on="idx", right_on="build_idx"
        )
        print(data)

        # apply compute_config_model_prediction_data_and_time to compute new columns
        tqdm.tqdm.pandas(desc="my bar!")
        model_pred_data = data.progress_apply(
            compute_config_model_prediction_data_and_time, axis=1, result_type="expand"
        )
        data["latency_pred"] = model_pred_data[0]
        data["bram_pred"] = model_pred_data[1]
        data["sklearn_latency_infrence_time"] = model_pred_data[2]
        data["sklearn_bram_infrence_time"] = model_pred_data[3]

        data.to_csv(dse_csv_fp, index=False)
    else:
        data = pd.read_csv(dse_csv_fp)

    data_for_plotting_vitis = pd.DataFrame()
    data_for_plotting_sklearn = pd.DataFrame()

    data_for_plotting_vitis["idx"] = data["idx"]
    data_for_plotting_sklearn["idx"] = data["idx"]

    data_for_plotting_vitis["vitis_hls_latency"] = data["runtime_synth"]
    data_for_plotting_vitis["vitis_hls_bram"] = data["bram"]
    data_for_plotting_vitis["vitis_tool_runtime"] = data["vitis_tool_runtime"]

    data_for_plotting_sklearn["latency_pred"] = data["latency_pred"]
    data_for_plotting_sklearn["bram_pred"] = data["bram_pred"]
    data_for_plotting_sklearn["sklearn_latency_infrence_time"] = data[
        "sklearn_latency_infrence_time"
    ]
    data_for_plotting_sklearn["sklearn_bram_infrence_time"] = data[
        "sklearn_bram_infrence_time"
    ]

    # shuffle rows
    # df.iloc[np.random.permutation(len(df))]
    # data_for_plotting_vitis_shuffled = data_for_plotting_vitis.iloc[np.random.permutation(len(data_for_plotting_vitis))].copy()
    # data_for_plotting_sklearn_shuffled = data_for_plotting_sklearn.iloc[np.random.permutation(len(data_for_plotting_sklearn))].copy()

    # data_for_plotting_vitis_shuffled["vitis_tool_runtime_cumsum"] = data_for_plotting_vitis_shuffled["vitis_tool_runtime"].cumsum()
    # data_for_plotting_sklearn_shuffled["sklearn_latency_infrence_time_cumsum"] = data_for_plotting_sklearn_shuffled["sklearn_latency_infrence_time"].cumsum()
    # data_for_plotting_sklearn_shuffled["sklearn_bram_infrence_time_cumsum"] = data_for_plotting_sklearn_shuffled["sklearn_bram_infrence_time"].cumsum()

    data_for_plotting_vitis["vitis_tool_runtime_cumsum"] = data_for_plotting_vitis[
        "vitis_tool_runtime"
    ].cumsum()
    data_for_plotting_sklearn["sklearn_latency_infrence_time_cumsum"] = (
        data_for_plotting_sklearn["sklearn_latency_infrence_time"].cumsum()
    )
    data_for_plotting_sklearn["sklearn_bram_infrence_time_cumsum"] = (
        data_for_plotting_sklearn["sklearn_bram_infrence_time"].cumsum()
    )

    average_vitis_tool_runtime = data_for_plotting_vitis["vitis_tool_runtime"].mean()
    average_sklearn_latency_infrence_time = data_for_plotting_sklearn[
        "sklearn_latency_infrence_time"
    ].mean()

    center_vitis_tool_runtime_cumsum = (
        data_for_plotting_vitis["vitis_tool_runtime_cumsum"].mean() / 4
    )
    center_sklearn_latency_infrence_time_cumsum = (
        data_for_plotting_sklearn["sklearn_latency_infrence_time_cumsum"].max() / 4
    )

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))

    font = {"size": 12}
    matplotlib.rc("font", **font)

    axs[0].scatter(
        data_for_plotting_vitis["vitis_tool_runtime_cumsum"],
        data_for_plotting_vitis["vitis_hls_latency"],
        label="vitis_hls_latency",
        marker="o",
        alpha=0.4,
        linewidths=0,
        s=10,
        color="#EE6983",
    )
    axs[0].scatter(
        data_for_plotting_sklearn["sklearn_latency_infrence_time_cumsum"],
        data_for_plotting_sklearn["latency_pred"] / 1e6,
        label="latency_pred",
        marker="o",
        alpha=0.4,
        linewidths=0,
        s=10,
        color="#1363DF",
    )

    # make x axis log scale but ticks should be seconds minutes hours etc
    axs[0].set_xlabel("Time (s)")
    axs[0].set_xscale("log")
    axs[0].set_xlim(1e-3, 1e6)
    # show minor ticks
    locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    axs[0].xaxis.set_minor_locator(locmin)
    ticks = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    axs[0].set_xticks(ticks)
    # make them small
    axs[0].tick_params(axis="both", which="major", labelsize=8)

    axs[0].axvline(x=0.010, color="red", linestyle="--", linewidth=1)
    axs[0].axvline(x=1, color="red", linestyle="--", linewidth=1)
    axs[0].axvline(x=60, color="red", linestyle="--", linewidth=1)
    axs[0].axvline(x=3600, color="red", linestyle="--", linewidth=1)
    axs[0].axvline(x=3600 * 24, color="red", linestyle="--", linewidth=1)
    axs[0].axvline(x=3600 * 24 * 7, color="red", linestyle="--", linewidth=1)

    trans = transforms.blended_transform_factory(axs[0].transData, axs[0].transAxes)

    axs[0].text(
        0.010, 1.02, "10ms", color="red", fontsize=8, ha="center", transform=trans
    )
    axs[0].text(1, 1.02, "1s", color="red", fontsize=8, ha="center", transform=trans)
    axs[0].text(60, 1.02, "1m", color="red", fontsize=8, ha="center", transform=trans)
    axs[0].text(3600, 1.02, "1h", color="red", fontsize=8, ha="center", transform=trans)
    axs[0].text(
        3600 * 24, 1.02, "1d", color="red", fontsize=8, ha="center", transform=trans
    )
    axs[0].text(
        3600 * 24 * 7, 1.02, "1w", color="red", fontsize=8, ha="center", transform=trans
    )

    axs[0].text(
        center_vitis_tool_runtime_cumsum,
        0.06,
        f"Avg. Runtime\n{average_vitis_tool_runtime:.2f} s",
        transform=trans,
        ha="center",
        fontsize=6,
        bbox=dict(facecolor="white", boxstyle="square,pad=0.2", lw=0),
    )
    axs[0].text(
        center_sklearn_latency_infrence_time_cumsum,
        0.06,
        f"Avg. Runtime\n{average_sklearn_latency_infrence_time:.4f} s",
        transform=trans,
        ha="center",
        fontsize=6,
        bbox=dict(facecolor="white", boxstyle="square,pad=0.2", lw=0),
    )

    axs[0].set_ylabel("Latency (s.)")
    # y axis is log scale from .01 ms to 10s
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-5, 1e-1)
    axs[0].set_yticklabels(["0.01ms", "0.1ms", "1ms", "10ms", "0.1s"])
    axs[0].set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

    vitis_scatter = axs[1].scatter(
        data_for_plotting_vitis["vitis_tool_runtime_cumsum"],
        data_for_plotting_vitis["vitis_hls_bram"],
        label="vitis_hls_bram",
        marker="o",
        alpha=0.4,
        linewidths=0,
        s=10,
        color="#EE6983",
    )
    sklearn_scatter = axs[1].scatter(
        data_for_plotting_sklearn["sklearn_bram_infrence_time_cumsum"],
        data_for_plotting_sklearn["bram_pred"],
        label="bram_pred",
        marker="o",
        alpha=0.4,
        linewidths=0,
        s=10,
        color="#1363DF",
    )

    axs[1].set_xscale("log")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_xlim(1e-3, 1e6)
    # show minor ticks
    locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    axs[1].xaxis.set_minor_locator(locmin)
    ticks = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    axs[1].set_xticks(ticks)
    # make them small
    axs[1].tick_params(axis="both", which="major", labelsize=8)

    axs[1].axvline(x=0.010, color="red", linestyle="--", linewidth=1)
    axs[1].axvline(x=1, color="red", linestyle="--", linewidth=1)
    axs[1].axvline(x=60, color="red", linestyle="--", linewidth=1)
    axs[1].axvline(x=3600, color="red", linestyle="--", linewidth=1)
    axs[1].axvline(x=3600 * 24, color="red", linestyle="--", linewidth=1)
    axs[1].axvline(x=3600 * 24 * 7, color="red", linestyle="--", linewidth=1)

    trans = transforms.blended_transform_factory(axs[1].transData, axs[1].transAxes)

    axs[1].text(
        0.010, 1.02, "10ms", color="red", fontsize=8, ha="center", transform=trans
    )
    axs[1].text(1, 1.02, "1s", color="red", fontsize=8, ha="center", transform=trans)
    axs[1].text(60, 1.02, "1m", color="red", fontsize=8, ha="center", transform=trans)
    axs[1].text(3600, 1.02, "1h", color="red", fontsize=8, ha="center", transform=trans)
    axs[1].text(
        3600 * 24, 1.02, "1d", color="red", fontsize=8, ha="center", transform=trans
    )
    axs[1].text(
        3600 * 24 * 7, 1.02, "1w", color="red", fontsize=8, ha="center", transform=trans
    )

    axs[1].set_ylabel("BRAM Count")
    axs[1].set_ylim(0, 5000)

    fig.suptitle("DSE of Estimated Latency and BRAM", y=0.95, fontsize=14)
    fig.legend(
        [sklearn_scatter, vitis_scatter],
        ["Direct Fit Model", "Vitis HLS"],
        loc="lower center",
        shadow=False,
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
        scatterpoints=4,
    )

    plt.tight_layout(rect=[0, 0, 1.0, 1])

    figure_dir_fp = Path("./figures/")
    os.makedirs(figure_dir_fp, exist_ok=True)
    plt.savefig(figure_dir_fp / "dse_plot.png", dpi=300)
