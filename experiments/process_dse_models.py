import pickle
import random
from typing import Any
import matplotlib
from matplotlib.offsetbox import bbox_artist
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from joblib import Parallel, delayed

from pathlib import Path
import os
from pprint import pp
import json
import copy

from torch_geometric.datasets import QM9

import gnn_builder as gnnb

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    HuberRegressor,
    RidgeCV,
    RANSACRegressor,
    SGDRegressor,
    LarsCV,
    LassoLarsCV,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    KFold,
)
from sklearn.compose import TransformedTargetRegressor

from sklearn.metrics import make_scorer


np.random.seed(0)
random.seed(0)

QM9_DATASET = QM9(root="./tmp/QM9").index_select(list(range(1000)))
DATASET_IN_DIM = QM9_DATASET.num_features
DATASET_OUT_DIM = QM9_DATASET[0].y.ravel().shape[0]
MEDIAN_NODES, MEDIAN_EDGES = gnnb.utils.compute_median_nodes_and_edges(
    QM9_DATASET, round_val=True
)
MEDIAN_DEGREE = gnnb.utils.compute_median_degree(QM9_DATASET)


def extract_single(idx, config, BUILD_DIR: Path):
    # for idx, config in zip(idxs, configs):
    perf_data_single: dict[str, Any] = {}

    proj_fp = BUILD_DIR / f"perf_model_{idx}_proj"
    proj_synth_rpt_fp = (
        proj_fp
        / f"perf_model_{idx}_proj_vitis_hls_project"
        / "solution1"
        / "syn"
        / "report"
        / f"perf_model_{idx}_proj_top_csynth.xml"
    )

    # if file does not exist, skip
    if not proj_synth_rpt_fp.exists():
        print(f"{proj_synth_rpt_fp} does not exist")
        runtime_synth = None
        bram = None
        dsp = None
    else:
        synth_data = gnnb.code_gen.extract_data_from_csynth_report(proj_synth_rpt_fp)
        runtime_synth = synth_data["worst_case_runtime"]
        bram = synth_data["bram"]
        dsp = synth_data["dsp"]

    perf_data_single = {
        "idx": idx,
        "gnn_in_dim": DATASET_IN_DIM,
        "gnn_hidden_dim": config["gnn_hidden_dim"],
        "gnn_out_dim": config["gnn_out_dim"],
        "gnn_num_layers": config["gnn_num_layers"],
        "mlp_in_dim": config["gnn_out_dim"] * 3,
        "mlp_hidden_dim": config["mlp_hidden_dim"],
        "mlp_out_dim": DATASET_OUT_DIM,
        "mlp_num_layers": config["mlp_num_layers"],
        "gnn_p_in": 1,
        "gnn_p_hidden": config["gnn_p_hidden"],
        "gnn_p_out": config["gnn_p_out"],
        "mlp_p_in": config["mlp_p_in"],
        "mlp_p_hidden": config["mlp_p_hidden"],
        "mlp_p_out": 1,
        "conv": config["conv"],
        "gnn_skip_connections": config["gnn_skip_connections"],
        "median_nodes": MEDIAN_NODES,
        "median_edges": MEDIAN_EDGES,
        "median_degree": MEDIAN_DEGREE,
        "runtime_synth": runtime_synth,
        "bram": bram,
        "dsp": dsp,
    }

    # perf_data_table.append(perf_data_single)
    return perf_data_single


def extract_perf_data(DATA_DIR: Path, BUILD_DIR: Path):
    configs = []
    idxs = []
    config_fps = DATA_DIR.glob("*config.json")
    config_fps_sorted = sorted(config_fps)
    for config_fp in config_fps_sorted:
        idx = int(config_fp.stem.split("_")[-2])
        idxs.append(idx)
        with open(config_fp, "r") as f:
            configs.append(json.load(f))

    perf_data_table: list[dict] = []

    # joblib to parallelize
    perf_data_table = Parallel(n_jobs=32, verbose=11, backend="multiprocessing")(
        delayed(extract_single)(idx, config, BUILD_DIR)
        for idx, config in tqdm.tqdm(zip(idxs, configs))
    )

    df = pd.DataFrame(perf_data_table)
    df.to_csv(DATA_DIR / "perf_data.csv", index=False)


def transform_x(x):
    x = x.copy()

    x["gnn_skip_connections"] = x["gnn_skip_connections"].astype(int)

    CONVS = [["gcn", "sage", "gin", "pna"]]
    one_hot_encoder = OneHotEncoder(sparse=False, categories=CONVS)
    conv_one_hot = one_hot_encoder.fit_transform(x["conv"].values.reshape(-1, 1))
    catagories = one_hot_encoder.categories_[0].tolist()
    catagories = [f"conv_{x}" for x in catagories]
    one_hot_df = pd.DataFrame(conv_one_hot, columns=catagories)

    x = pd.concat([x, one_hot_df], axis=1)
    x = x.drop(columns=["conv"])

    return x


if __name__ == "__main__":
    PERF_DATA_DATA_DIR = Path("./results_perf/")
    PERF_DATA_BUILD_DIR = Path("/usr/scratch/skaram7/gnnb_perf_builds/")

    csv_fp = PERF_DATA_DATA_DIR / "perf_data.csv"
    if not csv_fp.exists():
        extract_perf_data(PERF_DATA_DATA_DIR, PERF_DATA_BUILD_DIR)

    df = pd.read_csv(csv_fp)
    print(df.head())

    # drop any rows with NaN
    df = df.dropna(axis=0)
    df = df.reset_index()
    # print(df.shape)

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
    y_synth_col = "runtime_synth"

    numeric_features = [
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
        "median_nodes",
        "median_edges",
        "median_degree",
    ]
    # numeric_transformer = Pipeline()

    x = df[x_cols]
    x = transform_x(x)

    reg = Pipeline(
        # [("poly", PolynomialFeatures(degree=1)), ("lin", HuberRegressor(alpha=1e-2, max_iter=5000))]
        # [("poly", PolynomialFeatures(degree=1)), ("lin", TransformedTargetRegressor(HuberRegressor(max_iter=5000), func=np.expm1, inverse_func=np.log1p))]
        # [("poly", PolynomialFeatures(degree=1)), ("lin_kernel", KernelRidge(kernel="rbf", alpha=1, gamma=1e-2))]
        # [
        #     ("poly", PolynomialFeatures(degree=3, interaction_only=True)),
        #     ("tree", RandomForestRegressor(n_estimators=20)),
        # ],
        [
            ("poly", PolynomialFeatures(degree=1)),
            ("tree", ExtraTreesRegressor(n_estimators=10, max_depth=8)),
        ]
        # [("poly", PolynomialFeatures(degree=1)), ("gb", GradientBoostingRegressor(n_estimators=20))]
        # [("poly", PolynomialFeatures(degree=2)), ("knn", KNeighborsRegressor(n_neighbors=10, weights='distance', metric='cosine'))]
        # [
        #     ("poly", PolynomialFeatures(degree=1)),
        #     (
        #         "mlp",
        #         MLPRegressor(
        #             hidden_layer_sizes=(32, 32),
        #             solver="adam",
        #             learning_rate="invscaling",
        #             batch_size=128,
        #             activation="relu",
        #             max_iter=5000,
        #             n_iter_no_change=20,
        #             verbose=False,
        #         ),
        #     ),
        # ]
    )

    def mape(y, y_pred):
        return np.mean(np.abs((y - y_pred) / y)) * 100

    def mae(y, y_pred):
        return np.mean(np.abs(y - y_pred))

    y_synth = df[y_synth_col] * 1e6

    model_direct = copy.deepcopy(reg)
    model_direct.fit(x, y_synth)
    y_pred_direct = model_direct.predict(x)

    mape_fit_direct = mape(y_synth, y_pred_direct)
    print(f"MAPE fit direct: {mape_fit_direct}")

    scorer = make_scorer(mape, greater_is_better=False)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def eval_cross_validation(model, x, y, metric=mape):
        out = []
        preds = []
        for train_index, val_index in kf.split(x):
            model_ = copy.deepcopy(reg)
            x_train, x_val = x.iloc[train_index], x.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            model_.fit(x_train, y_train)

            y_pred = model_.predict(x_val)
            preds.append(y_pred)
            m = metric(y_val, y_pred)
            out.append(m)
        preds = np.concatenate(preds)
        return out, preds

    model_direct_scores, model_direct_all_pred = eval_cross_validation(
        model_direct, x, y_synth
    )
    model_direct_scores_avg = np.mean(model_direct_scores)
    print(f"CV Avg MAPE direct: {model_direct_scores_avg}")

    # print(model_direct_all_pred)
    # print(model_direct_all_pred.shape)

    y_bram = df["bram"]
    print(y_bram)

    model_bram = copy.deepcopy(reg)
    model_bram.fit(x, y_bram)
    y_pred_bram = model_bram.predict(x)
    mape_fit_bram = mape(y_bram, y_pred_bram)
    print(f"MAPE fit bram: {mape_fit_bram}")

    scores, model_bram_all_pred = eval_cross_validation(model_bram, x, y_bram)
    mape_fit_bram_scores_avg = np.mean(scores)
    print(f"CV Avg MAPE bram: {mape_fit_bram_scores_avg}")

    # fig, axd = plt.subplot_mosaic(
    #     [["upper left", "lower left"], ["upper right", "lower right"]], figsize=(8, 7), constrained_layout=False
    # )

    font = {"size": 10}
    matplotlib.rc("font", **font)

    fig, axd = plt.subplot_mosaic(
        [["left", "right"]], figsize=(7, 3), constrained_layout=False
    )

    # axd["upper left"].scatter(y_perf, y_synth, marker="o", alpha=0.8, linewidths=0, s=10, c="#be95c4")
    # axd["upper left"].plot([y_synth.min(), y_synth.max()], [y_synth.min(), y_synth.max()], "k--", lw=1)
    # axd["upper left"].set_xlim(y_synth.min(), y_synth.max())
    # axd["upper left"].set_ylim(y_synth.min(), y_synth.max())
    # axd["upper left"].set_xscale("log")
    # axd["upper left"].set_yscale("log")
    # axd["upper left"].set_xlabel("Model Predicted Latency (s)")
    # axd["upper left"].set_ylabel("Synth. Predicted Latency (s)")
    # axd["upper left"].set_title("Analytical Perf. Model vs.\nHLS Synth. Predicted Latency")
    # # show mape in upper left corner
    # axd["upper left"].text(
    #     0.05,
    #     0.9,
    #     f"MAPE: {mape_perf:.2f}%",
    #     horizontalalignment="left",
    #     verticalalignment="center",
    #     transform=axd["upper left"].transAxes,
    #     bbox=dict(fc="w", ec="k", pad=4),
    # )

    # axd["upper right"].scatter(y_pred_direct, y_synth, marker="o", alpha=0.8, linewidths=0, s=10, c="#be95c4")
    # axd["upper right"].plot([y_synth.min(), y_synth.max()], [y_synth.min(), y_synth.max()], "k--", lw=1)
    # axd["upper right"].set_xlim(y_synth.min(), y_synth.max())
    # axd["upper right"].set_ylim(y_synth.min(), y_synth.max())
    # axd["upper right"].set_xscale("log")
    # axd["upper right"].set_yscale("log")
    # axd["upper right"].set_xlabel("Model Predicted Latency (s)")
    # axd["upper right"].set_ylabel("Synth. Predicted Latency (s)")
    # axd["upper right"].set_title("Direct Fit Model vs.\nHLS Synth. Predicted Latency")
    # # show mape in bottom right corner
    # axd["upper right"].text(
    #     0.05,
    #     0.9,
    #     f"Avg. CV MAPE: {model_direct_scores_avg:.2f}%",
    #     horizontalalignment="left",
    #     verticalalignment="center",
    #     transform=axd["upper right"].transAxes,
    #     bbox=dict(fc="w", ec="k", pad=4),
    # )

    axd["left"].scatter(
        y_pred_direct / 1e6,
        y_synth / 1e6,
        marker="o",
        alpha=0.8,
        linewidths=0,
        s=10,
        c="#be95c4",
    )
    axd["left"].plot(
        [(y_synth / 1e6).min(), (y_synth / 1e6).max()],
        [(y_synth / 1e6).min(), (y_synth / 1e6).max()],
        "k--",
        lw=1,
    )
    axd["left"].set_xlim((y_synth / 1e6).min(), (y_synth / 1e6).max())
    axd["left"].set_ylim((y_synth / 1e6).min(), (y_synth / 1e6).max())
    axd["left"].set_xscale("log")
    axd["left"].set_yscale("log")
    axd["left"].set_xlabel("Model Predicted Latency (s)")
    axd["left"].set_ylabel("Synth. Predicted Latency (s)")
    axd["left"].set_title("Direct-Fit Latency Model vs.\nHLS Synth. Predicted Latency")
    # show mape in bottom right corner
    axd["left"].text(
        0.05,
        0.9,
        f"Avg. CV MAPE: {model_direct_scores_avg:.2f}%",
        horizontalalignment="left",
        verticalalignment="center",
        transform=axd["left"].transAxes,
        bbox=dict(fc="w", ec="k", pad=4),
    )

    # axd["lower left"].scatter(y_pred_residual, y_synth, marker="o", alpha=0.8, linewidths=0, s=10, c="#be95c4")
    # axd["lower left"].plot([y_synth.min(), y_synth.max()], [y_synth.min(), y_synth.max()], "k--", lw=1)
    # axd["lower left"].set_xlim(y_synth.min(), y_synth.max())
    # axd["lower left"].set_ylim(y_synth.min(), y_synth.max())
    # axd["lower left"].set_xscale("log")
    # axd["lower left"].set_yscale("log")
    # axd["lower left"].set_xlabel("Model Predicted Latency (s)")
    # axd["lower left"].set_ylabel("Synth. Predicted Latency (s)")
    # axd["lower left"].set_title("Analytical  Model + Compensation Fit vs.\nHLS Synth. Predicted Latency")
    # # show mape in bottom right corner
    # axd["lower left"].text(
    #     0.05,
    #     0.9,
    #     f"Avg. CV MAPE: {model_residual_scores_avg:.2f}%",
    #     horizontalalignment="left",
    #     verticalalignment="center",
    #     transform=axd["lower left"].transAxes,
    #     bbox=dict(fc="w", ec="k", pad=4),
    # )

    # axd["lower right"].scatter(y_pred_bram, y_bram, marker="o", alpha=0.8, linewidths=0, s=10, c="#8d99ae")
    # axd["lower right"].plot([y_bram.min(), y_bram.max()], [y_bram.min(), y_bram.max()], "k--", lw=1)
    # axd["lower right"].set_xlim(y_bram.min(), y_bram.max())
    # axd["lower right"].set_ylim(y_bram.min(), y_bram.max())
    # # ax.set_xscale("log")
    # # ax.set_yscale("log")
    # axd["lower right"].set_xlabel("Model Predicted BRAM Count")
    # axd["lower right"].set_ylabel("Synth. Predicted BRAM Count")
    # axd["lower right"].set_title("Direct Fit Model vs.\nHLS Synth. Predicted BRAM Count")
    # # show mape in bottom right corner
    # axd["lower right"].text(
    #     0.05,
    #     0.9,
    #     f"Avg. CV MAPE: {mape_fit_bram_scores_avg:.2f}%",
    #     horizontalalignment="left",
    #     verticalalignment="center",
    #     transform=axd["lower right"].transAxes,
    #     bbox=dict(fc="w", ec="k", pad=4),
    # )

    axd["right"].scatter(
        y_pred_bram, y_bram, marker="o", alpha=0.8, linewidths=0, s=10, c="#8d99ae"
    )
    axd["right"].plot(
        [y_bram.min(), y_bram.max()], [y_bram.min(), y_bram.max()], "k--", lw=1
    )
    axd["right"].set_xlim(y_bram.min(), y_bram.max())
    axd["right"].set_ylim(y_bram.min(), y_bram.max())
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    axd["right"].set_xlabel("Model Predicted BRAM Count")
    axd["right"].set_ylabel("Synth. Predicted BRAM Count")
    axd["right"].set_title("Direct-Fit BRAM Model vs.\nHLS Synth. Predicted BRAM Count")
    # show mape in bottom right corner
    axd["right"].text(
        0.05,
        0.9,
        f"Avg. CV MAPE: {mape_fit_bram_scores_avg:.2f}%",
        horizontalalignment="left",
        verticalalignment="center",
        transform=axd["right"].transAxes,
        bbox=dict(fc="w", ec="k", pad=4),
    )

    fig.suptitle("Performance Models Comparison", fontsize=14, y=0.98)

    plt.tight_layout()
    plt.savefig("./figures/perf_data.png", dpi=300)

    DSE_MODEL_DIR = Path("./dse_models/")
    os.makedirs(DSE_MODEL_DIR, exist_ok=True)

    # latency models
    with open(DSE_MODEL_DIR / "model_direct.pk", "wb") as f:
        pickle.dump(model_direct, f)

    # bram models
    with open(DSE_MODEL_DIR / "model_bram.pk", "wb") as f:
        pickle.dump(model_bram, f)
