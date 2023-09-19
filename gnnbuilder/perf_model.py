import re

import torch
import tqdm
from rich.pretty import pprint as pp
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GINConv, PNAConv
from torch_geometric.utils import degree

from .models import (
    MLP,
    GATConv_GNNB,
    GCNConv_GNNB,
    GINConv_GNNB,
    GlobalPooling,
    GNNModel,
    PNAConv_GNNB,
    SAGEConv_GNNB,
)
