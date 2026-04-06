# graph_builder.py
# ─────────────────────────────────────────────────────────────────────────────
# Constructs a directed weighted transaction graph from PaySim data and
# extracts rich structural node features for GNN training.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import numpy as np
import pandas as pd
import networkx as nx
import joblib
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_networkx, to_undirected
from tqdm import tqdm
from typing import Tuple, Dict

from config import GRAPH_CONFIG, FEATURE_CONFIG, GRAPH_PKL, DATA_DIR, SEED

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build NetworkX Graph
# ─────────────────────────────────────────────────────────────────────────────

def build_networkx_graph(df: pd.DataFrame) -> Tuple[nx.DiGraph, Dict]:
    """
    Constructs a directed multi-edge transaction graph.

    Nodes  : unique account IDs (nameOrig ∪ nameDest)
    Edges  : transactions (directed: sender → receiver)
    Node attrs: fraud label (1 if ever sent a fraudulent transaction)
    Edge attrs : amount, step, type, isFraud

    Returns
    -------
    G          : nx.DiGraph
    node_map   : {account_id: integer_index}
    """
    log.info("Building NetworkX transaction graph …")

    # Optionally cap graph size
    max_nodes = GRAPH_CONFIG["max_graph_nodes"]
    if len(df) > max_nodes:
        log.warning(f"  Sampling {max_nodes} transactions for graph (memory cap).")
        # Oversample fraud to keep them in the graph
        fraud   = df[df["isFraud"] == 1]
        normal  = df[df["isFraud"] == 0].sample(
            n=min(max_nodes - len(fraud), len(df[df["isFraud"]==0])),
            random_state=SEED
        )
        df = pd.concat([fraud, normal]).reset_index(drop=True)

    # Build account → integer index
    all_accounts = pd.concat([df["nameOrig"], df["nameDest"]]).unique()
    node_map = {acct: idx for idx, acct in enumerate(all_accounts)}
    log.info(f"  Nodes: {len(node_map):,}  Edges: {len(df):,}")

    G = nx.DiGraph()

    # Add nodes with fraud label
    fraud_accounts = set(df[df["isFraud"] == 1]["nameOrig"].unique())
    for acct, idx in node_map.items():
        G.add_node(idx,
                   account_id=acct,
                   is_fraud=int(acct in fraud_accounts))

    # Add edges
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Adding edges"):
        src = node_map[row["nameOrig"]]
        dst = node_map[row["nameDest"]]
        G.add_edge(src, dst,
                   amount=float(row["amount"]),
                   amount_log=float(np.log1p(row["amount"])),
                   step=int(row["step"]),
                   tx_type=row["type_encoded"] if "type_encoded" in row else 0,
                   is_fraud=int(row["isFraud"]))

    log.info(f"  Graph built: {G.number_of_nodes():,} nodes, "
             f"{G.number_of_edges():,} edges")
    return G, node_map


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Structural Node Features
# ─────────────────────────────────────────────────────────────────────────────

def compute_graph_features(G: nx.DiGraph) -> pd.DataFrame:
    """
    Computes structural graph features for each node.

    Features
    --------
    - in_degree, out_degree, degree
    - weighted_in_degree, weighted_out_degree
    - pagerank
    - betweenness_centrality (approx for large graphs)
    - clustering_coefficient
    - avg_neighbor_degree
    - local_fraud_rate  (fraction of neighbors that are fraud)
    """
    log.info("Computing structural node features …")
    n = G.number_of_nodes()

    # Degree features
    log.info("  Degree …")
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    w_in    = dict(G.in_degree(weight="amount"))
    w_out   = dict(G.out_degree(weight="amount"))

    # PageRank
    log.info("  PageRank …")
    pr = nx.pagerank(G, alpha=0.85, max_iter=100, weight="amount")

    # Betweenness (approximate for large graphs)
    log.info("  Betweenness centrality (approx) …")
    k = min(500, n)
    bc = nx.betweenness_centrality(G, k=k, normalized=True, weight="amount",
                                   seed=SEED)

    # Clustering (on undirected version)
    log.info("  Clustering coefficient …")
    G_undir = G.to_undirected()
    cc = nx.clustering(G_undir)

    # Average neighbor degree
    log.info("  Avg neighbor degree …")
    avg_nd = nx.average_neighbor_degree(G)

    # Local fraud rate among neighbors
    log.info("  Local fraud rate …")
    fraud_labels = nx.get_node_attributes(G, "is_fraud")
    local_fraud = {}
    for node in G.nodes():
        neighbors = list(G.predecessors(node)) + list(G.successors(node))
        if neighbors:
            local_fraud[node] = np.mean([fraud_labels.get(nb, 0) for nb in neighbors])
        else:
            local_fraud[node] = 0.0

    # Assemble DataFrame
    rows = []
    for node in G.nodes():
        rows.append({
            "node_id"           : node,
            "in_degree"         : in_deg.get(node, 0),
            "out_degree"        : out_deg.get(node, 0),
            "degree"            : in_deg.get(node,0) + out_deg.get(node,0),
            "weighted_in_deg"   : w_in.get(node, 0),
            "weighted_out_deg"  : w_out.get(node, 0),
            "pagerank"          : pr.get(node, 0),
            "betweenness"       : bc.get(node, 0),
            "clustering_coeff"  : cc.get(node, 0),
            "avg_neighbor_deg"  : avg_nd.get(node, 0),
            "local_fraud_rate"  : local_fraud.get(node, 0),
            "is_fraud"          : fraud_labels.get(node, 0),
        })

    feat_df = pd.DataFrame(rows).set_index("node_id")
    log.info(f"  Graph features shape: {feat_df.shape}")
    return feat_df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PyG Data Object
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_FEAT_COLS = [
    "in_degree", "out_degree", "degree",
    "weighted_in_deg", "weighted_out_deg",
    "pagerank", "betweenness", "clustering_coeff",
    "avg_neighbor_deg", "local_fraud_rate",
]

def build_pyg_data(G: nx.DiGraph,
                   graph_feat_df: pd.DataFrame,
                   tabular_feats: np.ndarray,
                   node_map: dict,
                   df: pd.DataFrame) -> Data:
    """
    Converts the NetworkX graph + node features into a PyTorch Geometric
    Data object ready for GNN training.

    Node feature matrix X is the concatenation of:
      - Structural graph features (11 cols)
      - Tabular transaction-level aggregated features
    """
    log.info("Assembling PyG Data object …")
    n = G.number_of_nodes()

    # ── Node features ─────────────────────────────────────────────────────────
    # Structural (graph) features
    gf = graph_feat_df[GRAPH_FEAT_COLS].values.astype(np.float32)

    # Per-account aggregated tabular features
    # Aggregate by nameOrig to assign tabular features to each node
    AGGS = {
        "amount"          : "mean",
        "amount_log"      : "mean",
        "balance_diff_orig": "mean",
        "balance_diff_dest": "mean",
        "orig_tx_count"   : "first",
        "orig_total_sent" : "first",
        "orig_unique_dests": "first",
        "type_encoded"    : "mean",
    }
    # Only use columns that exist
    agg_cols = {k: v for k, v in AGGS.items() if k in df.columns}
    agg = df.groupby("nameOrig").agg(agg_cols).reset_index()
    agg = agg.rename(columns={"nameOrig": "account"})

    # Create node-level tabular feature matrix (zeros for destination-only nodes)
    tab_feat_dim = len(agg_cols)
    tab_node_feats = np.zeros((n, tab_feat_dim), dtype=np.float32)
    for _, row in agg.iterrows():
        if row["account"] in node_map:
            idx = node_map[row["account"]]
            tab_node_feats[idx] = [row[c] for c in agg_cols.keys()]

    # Concatenate structural + tabular
    x = np.concatenate([gf, tab_node_feats], axis=1)
    x = np.nan_to_num(x)  # safety

    # ── Edge index ────────────────────────────────────────────────────────────
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]

    # ── Edge attributes ───────────────────────────────────────────────────────
    edge_attrs = []
    for u, v, data in G.edges(data=True):
        edge_attrs.append([
            data.get("amount_log", 0),
            data.get("tx_type", 0),
            data.get("step", 0) / 744,      # normalize step
            data.get("is_fraud", 0),
        ])
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # ── Labels ────────────────────────────────────────────────────────────────
    y = torch.tensor(
        [G.nodes[i]["is_fraud"] for i in range(n)],
        dtype=torch.long
    )

    data = Data(
        x          = torch.tensor(x, dtype=torch.float),
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = y,
        num_nodes  = n,
    )
    log.info(f"  PyG Data: {data}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Temporal Subgraph (for real-time simulation)
# ─────────────────────────────────────────────────────────────────────────────

def get_temporal_subgraph(G: nx.DiGraph,
                          current_step: int,
                          window: int = 24) -> nx.DiGraph:
    """
    Returns the subgraph containing only edges within the past `window` steps.
    Used for real-time / streaming fraud detection.
    """
    edges_in_window = [
        (u, v) for u, v, d in G.edges(data=True)
        if current_step - window <= d.get("step", 0) <= current_step
    ]
    return G.edge_subgraph(edges_in_window).copy()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_pipeline(df: pd.DataFrame) -> Tuple[nx.DiGraph, Data, dict]:
    """
    Full graph construction pipeline.

    Returns
    -------
    G            : NetworkX DiGraph
    pyg_data     : PyTorch Geometric Data object
    graph_artifacts : dict with node_map and feature DataFrame
    """
    G, node_map = build_networkx_graph(df)
    graph_feat_df = compute_graph_features(G)
    pyg_data = build_pyg_data(G, graph_feat_df, None, node_map, df)

    artifacts = {
        "G"             : G,
        "node_map"      : node_map,
        "graph_feat_df" : graph_feat_df,
        "pyg_data"      : pyg_data,
    }
    joblib.dump(artifacts, GRAPH_PKL)
    log.info(f"Graph artifacts saved → {GRAPH_PKL}")
    return G, pyg_data, artifacts


if __name__ == "__main__":
    import pandas as pd
    from config import PROCESSED_CSV

    df = pd.read_csv(PROCESSED_CSV)
    # Use a subset for quick testing
    sample = pd.concat([
        df[df["isFraud"] == 1].head(5000),
        df[df["isFraud"] == 0].head(50000),
    ])
    G, pyg_data, artifacts = build_graph_pipeline(sample)
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"PyG: {pyg_data}")
    print(f"Fraud nodes: {(pyg_data.y == 1).sum().item()}")
