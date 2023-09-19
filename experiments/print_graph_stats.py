import pandas as pd
from torch_geometric.datasets import QM9, MoleculeNet

from gnnbuilder.utils import compute_median_nodes_and_edges

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

df = pd.DataFrame(
    columns=[
        "dataset",
        "median_nodes",
        "median_edges",
    ]
)

for d in DATASETS:
    median_nodes, median_edges = compute_median_nodes_and_edges(DATASETS[d])
    print(f"{d}: median_nodes={median_nodes}, median_edges={median_edges}")
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "dataset": [d],
                    "median_nodes": [median_nodes],
                    "median_edges": [median_edges],
                }
            ),
        ]
    )


latex_table = df.to_latex(
    index=False,
    column_format="lrr",
    escape=False,
    bold_rows=True,
    label="tab:dataset_stats",
    caption="Median number of nodes and edges in the datasets.",
    header=["Dataset", "Median # of Nodes", "Median # of Edges"],
)

print(latex_table)
