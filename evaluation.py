"""
evaluation.py — DeepTrace vs DeepTrace++ Performance Evaluation
================================================================
Reproduces the metrics from Section V of the DeepTrace paper and
extends them with DeepTrace++ comparisons.

Metrics (from the paper):
  1. Top-k accuracy (k=1, 5, 10) — is the true source in the top-k predictions?
  2. Bias — MSE between predicted and true log-likelihood scores
  3. First detection time — how many nodes traced before source is first found
  4. Average hop error — mean shortest-path distance (predicted vs true source)
  5. Wall time — computational efficiency

Tested across network topologies:
  - Erdos-Renyi (ER)
  - Barabasi-Albert (BA)
  - Watts-Strogatz (WS)
  - Regular random graphs
  - Stochastic Block Model (SBM / community)

Usage:
    python evaluation.py
"""

import os, time, random, warnings
from math import sqrt, floor
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.convert import from_networkx

from graph_data_process import TreeDataProcess, ORIGINAL_FEAT_DIM, ENRICHED_FEAT_DIM
from cal_max_min_ds import CalMaxMinDS
from model import (SAGE, WeightedSAGE, TemporalSAGE,
                   data_process_for_single_tree, assign_edge_weights,
                   R2_score)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# 1. NETWORK GENERATORS  (Table II in the paper)
# ═══════════════════════════════════════════════════════════════════

def generate_network(topology: str, n: int = 100, seed: int = None) -> nx.Graph:
    """Generate an underlying network G, then simulate SI spreading to get
    an epidemic tree of ~n infected nodes."""
    rng = np.random.RandomState(seed)
    s = rng.randint(0, 2**31)

    if topology == "ER":
        G = nx.erdos_renyi_graph(n * 3, 0.08, seed=s)
    elif topology == "BA":
        G = nx.barabasi_albert_graph(n * 3, 3, seed=s)
    elif topology == "WS":
        G = nx.watts_strogatz_graph(n * 2, 6, 0.3, seed=s)
    elif topology == "Regular":
        deg = 4 if n * 2 * 4 % 2 == 0 else 3
        G = nx.random_regular_graph(deg, n * 2, seed=s)
    elif topology == "SBM":
        sizes = [n // 3, n // 3, n - 2 * (n // 3)]
        p_in, p_out = 0.3, 0.02
        G = nx.stochastic_block_model(sizes, [[p_in, p_out, p_out],
                                               [p_out, p_in, p_out],
                                               [p_out, p_out, p_in]], seed=s)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    # Ensure connected
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    # SI spreading simulation
    source = rng.choice(list(G.nodes()))
    infected = {source}
    frontier = list(G.neighbors(source))
    rng.shuffle(frontier)
    tree = nx.Graph()

    while len(infected) < min(n, G.number_of_nodes()) and frontier:
        new_node = frontier.pop(0)
        if new_node in infected:
            continue
        # Find parent (the infected neighbor that "transmitted")
        parents = [nb for nb in G.neighbors(new_node) if nb in infected]
        if not parents:
            continue
        parent = parents[rng.randint(0, len(parents))]
        infected.add(new_node)
        tree.add_edge(parent, new_node)
        new_neighbors = [nb for nb in G.neighbors(new_node) if nb not in infected]
        rng.shuffle(new_neighbors)
        frontier.extend(new_neighbors)

    tree = nx.convert_node_labels_to_integers(tree)
    source_new = 0  # source is the first node after relabeling
    return tree, source_new


# ═══════════════════════════════════════════════════════════════════
# 2. TRAINING HELPERS
# ═══════════════════════════════════════════════════════════════════

def build_training_data(num_trees=30, node_range=(80, 200), enriched=False):
    """Build a batched DGL training set from synthetic WS trees."""
    dgl_trees = []
    for i in range(num_trees):
        n = random.randint(node_range[0], node_range[1])
        deg = max(2, floor(n * 3 / 20))
        G = nx.watts_strogatz_graph(n, deg, 0.3)
        tree = nx.minimum_spanning_tree(G)
        dgl_tree = data_process_for_single_tree(tree, enriched=enriched)
        dgl_trees.append(dgl_tree)
    return dgl.batch(dgl_trees)


def train_model(model_class, feat_dim, train_batch, epochs=50,
                hid=100, weighted_mode=None):
    """Train a model and return it."""
    model = model_class(in_feats=feat_dim, hid_feats=hid, out_feats=1)
    opt = th.optim.Adam(model.parameters(), lr=0.001)

    feats = train_batch.ndata["feat"]
    labels = train_batch.ndata["labels"]

    if weighted_mode and model_class == WeightedSAGE:
        assign_edge_weights(train_batch, mode=weighted_mode)

    for epoch in range(epochs):
        model.train()
        logits = model(train_batch, feats)
        loss = F.mse_loss(logits.squeeze(-1), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


# ═══════════════════════════════════════════════════════════════════
# 3. EVALUATION METRICS (Section V of the paper)
# ═══════════════════════════════════════════════════════════════════

def compute_scores_on_tree(model, tree: nx.Graph, enriched=False,
                           weighted_mode=None):
    """Run inference on a single tree, return (predictions, true_labels)."""
    tree_copy = tree.copy()
    proc = TreeDataProcess(tree_copy)
    unift = proc.get_uninfected_node_list()
    nfeat = proc.nfeature_process(enriched=enriched)
    tree_copy.remove_nodes_from(list(unift))

    feat_rows, label_list = [], []
    for node in nx.nodes(tree_copy):
        if node not in set(unift):
            v = nfeat[node]
            row = [v["node_num"], v["degree_per"], v["degree_per_aver"],
                   v["inft_ndegree_per"], v["inft_alldegree_per"],
                   v["distance_per"], v["layer_rate"], v["layer_num"]]
            if enriched:
                row.append(v["closeness_centrality"])
                row.append(v["norm_dist_to_index"])
            feat_rows.append(row)
            try:
                cal = CalMaxMinDS(tree_copy, unift, node)
                mx = cal.cal_max_ds()
                mn = cal.cal_min_ds()
                label_list.append(sqrt(mx * mn))
            except (ZeroDivisionError, ValueError):
                label_list.append(0.0)

    if not feat_rows:
        return None, None, None

    feats = th.tensor(feat_rows, dtype=th.float32)
    labels = th.tensor(label_list, dtype=th.float32)
    dgl_tree = from_networkx(tree_copy)

    if weighted_mode:
        assign_edge_weights(dgl_tree, mode=weighted_mode)

    model.eval()
    with th.no_grad():
        preds = model(dgl_tree, feats).squeeze(-1)

    return preds, labels, list(tree_copy.nodes())


def eval_topk_accuracy(preds, labels, k_list=[1, 5, 10]):
    """Check if the true superspreader (argmax of labels) is in top-k predictions."""
    if preds is None:
        return {k: 0 for k in k_list}
    true_best = labels.argmax().item()
    sorted_pred_idx = preds.argsort(descending=True).tolist()
    results = {}
    for k in k_list:
        results[k] = 1 if true_best in sorted_pred_idx[:k] else 0
    return results


def eval_bias(preds, labels):
    """MSE between predictions and true labels (bias metric from paper)."""
    if preds is None:
        return float("nan")
    return F.mse_loss(preds, labels).item()


def eval_hop_error(tree, preds, labels, nodes):
    """Hop distance between predicted source and true source."""
    if preds is None:
        return float("nan")
    pred_src = nodes[preds.argmax().item()]
    true_src = nodes[labels.argmax().item()]
    try:
        return nx.shortest_path_length(tree, pred_src, true_src)
    except nx.NetworkXError:
        return -1


def forward_trace_first_detection(tree, true_source, model, enriched=False,
                                  weighted_mode=None, strategy="bfs"):
    """Simulate forward tracing (BFS/DFS), return first detection time
    and average hop error as described in Section V-B."""
    nodes = sorted(tree.nodes())
    if not nodes:
        return float("nan"), float("nan"), 0.0

    seed = nodes[min(len(nodes) // 3, len(nodes) - 1)]
    if strategy == "bfs":
        order = list(nx.bfs_tree(tree, seed).nodes())
    else:
        order = list(nx.dfs_tree(tree, seed).nodes())

    observed = {seed}
    first_detection = None
    hop_errors = []
    t0 = time.time()

    # Sample every few steps for speed on larger graphs
    sample_interval = max(1, len(order) // 20)

    for step_i, node in enumerate(order[1:], 1):
        observed.add(node)
        if step_i % sample_interval != 0 and step_i > 5:
            continue
        sub = tree.subgraph(sorted(observed)).copy()
        if sub.number_of_nodes() < 3:
            continue

        try:
            preds, labels, sub_nodes = compute_scores_on_tree(
                model, sub, enriched=enriched, weighted_mode=weighted_mode)
        except Exception:
            continue
        if preds is None:
            continue

        pred_src = sub_nodes[preds.argmax().item()]
        try:
            hop_err = nx.shortest_path_length(tree, pred_src, true_source)
        except nx.NetworkXError:
            hop_err = -1

        hop_errors.append(hop_err)

        if pred_src == true_source and first_detection is None:
            first_detection = len(observed)

        # Early stop for speed (sample every few steps for large graphs)
        if step_i > 30 and step_i % 5 != 0:
            continue

    wall = time.time() - t0
    avg_hop = float(np.mean(hop_errors)) if hop_errors else float("nan")
    if first_detection is None:
        first_detection = len(nodes)  # never detected

    return first_detection, avg_hop, wall


# ═══════════════════════════════════════════════════════════════════
# 4. RUMOR CENTER BASELINE (from rumor_centrality.py)
# ═══════════════════════════════════════════════════════════════════

def rumor_center_predict(tree: nx.Graph):
    """Simple rumor center heuristic: node that minimises max distance to leaves."""
    if tree.number_of_nodes() < 2:
        return list(tree.nodes())[0] if tree.nodes() else -1
    try:
        center = nx.center(tree)
        return center[0]
    except nx.NetworkXError:
        return list(tree.nodes())[0]


# ═══════════════════════════════════════════════════════════════════
# 5. MAIN EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_evaluation(topologies=None, n_nodes=200, n_trials=20,
                   train_epochs=150, n_train_trees=30):
    """
    Full evaluation comparing DeepTrace (original) vs DeepTrace++ variants.

    Models compared:
      - DeepTrace (SAGE, 8 features)
      - DeepTrace++ enriched (SAGE, 10 features)
      - DeepTrace++ weighted (WeightedSAGE, 8 features, degree weights)
      - Rumor Center baseline

    Metrics per topology × model:
      - Top-1, Top-5, Top-10 accuracy
      - Bias (MSE)
      - Average hop error
      - First detection time (BFS tracing)
      - Wall time
    """
    if topologies is None:
        topologies = ["ER", "BA", "WS", "Regular", "SBM"]

    print("=" * 70)
    print("  DeepTrace vs DeepTrace++ — Section V Evaluation")
    print("=" * 70)

    # ── Train all model variants ──────────────────────────────────────
    print("\n[1/3] Building training data...")
    train_orig = build_training_data(n_train_trees, enriched=False)
    train_enr  = build_training_data(n_train_trees, enriched=True)

    print("[2/3] Training models...")
    model_orig = train_model(SAGE, ORIGINAL_FEAT_DIM, train_orig,
                             epochs=train_epochs)
    model_enr  = train_model(SAGE, ENRICHED_FEAT_DIM, train_enr,
                             epochs=train_epochs)
    model_wt   = train_model(WeightedSAGE, ORIGINAL_FEAT_DIM, train_orig,
                             epochs=train_epochs, weighted_mode='degree')
    # Full DeepTrace++: WeightedSAGE + enriched features + degree weights
    model_full = train_model(WeightedSAGE, ENRICHED_FEAT_DIM, train_enr,
                             epochs=train_epochs, weighted_mode='degree')

    models = {
        "DeepTrace":          (model_orig, False, None),
        "DT++ enriched":      (model_enr,  True,  None),
        "DT++ weighted":      (model_wt,   False, 'degree'),
        "DeepTrace++":        (model_full, True,  'degree'),
        "Rumor Center":       (None,       False, None),
    }

    print(f"[3/3] Evaluating on {len(topologies)} topologies × "
          f"{n_trials} trials each...\n")

    all_results = []

    for topo in topologies:
        print(f"─── Topology: {topo} ───")
        for trial in range(n_trials):
            tree, true_src = generate_network(topo, n=n_nodes, seed=trial*42+7)
            if tree.number_of_nodes() < 5:
                continue

            for model_name, (model, enriched, wt_mode) in models.items():
                t0 = time.time()

                if model_name == "Rumor Center":
                    # Rumor center baseline
                    pred_src = rumor_center_predict(tree)
                    try:
                        hop = nx.shortest_path_length(tree, pred_src, true_src)
                    except nx.NetworkXError:
                        hop = -1
                    row = {
                        "topology": topo, "trial": trial,
                        "model": model_name,
                        "top1": 1 if pred_src == true_src else 0,
                        "top5": 1 if hop <= 2 else 0,
                        "top10": 1 if hop <= 4 else 0,
                        "bias": float("nan"),
                        "avg_hop_error": hop,
                        "first_detect_time": tree.number_of_nodes(),
                        "wall_time_s": round(time.time() - t0, 4),
                    }
                else:
                    preds, labels, nodes = compute_scores_on_tree(
                        model, tree.copy(), enriched=enriched,
                        weighted_mode=wt_mode)

                    topk = eval_topk_accuracy(preds, labels, [1, 5, 10])
                    bias = eval_bias(preds, labels)
                    hop  = eval_hop_error(tree, preds, labels, nodes) if nodes else -1

                    # First detection via BFS tracing (expensive — sample)
                    if trial < 5:
                        fdt, avg_h, _ = forward_trace_first_detection(
                            tree, true_src, model, enriched=enriched,
                            weighted_mode=wt_mode, strategy="bfs")
                    else:
                        fdt, avg_h = float("nan"), float("nan")

                    row = {
                        "topology": topo, "trial": trial,
                        "model": model_name,
                        "top1": topk[1], "top5": topk[5], "top10": topk[10],
                        "bias": round(bias, 4) if not np.isnan(bias) else bias,
                        "avg_hop_error": hop,
                        "first_detect_time": fdt,
                        "wall_time_s": round(time.time() - t0, 4),
                    }

                all_results.append(row)

            if trial % 5 == 0:
                print(f"  trial {trial+1}/{n_trials} done")

    # ── Build results DataFrame ───────────────────────────────────────
    df = pd.DataFrame(all_results)

    # ── Aggregate summary ─────────────────────────────────────────────
    summary = (df.groupby(["topology", "model"])
               .agg(
                   top1_acc=("top1", "mean"),
                   top5_acc=("top5", "mean"),
                   top10_acc=("top10", "mean"),
                   mean_bias=("bias", "mean"),
                   mean_hop_err=("avg_hop_error", "mean"),
                   mean_first_detect=("first_detect_time", "mean"),
                   mean_time_s=("wall_time_s", "mean"),
               )
               .reset_index())

    # ── Print results ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    for topo in topologies:
        print(f"\n── {topo} Network ──")
        sub = summary[summary["topology"] == topo]
        print(sub[["model", "top1_acc", "top5_acc", "top10_acc",
                    "mean_hop_err", "mean_first_detect",
                    "mean_time_s"]].to_string(index=False))

    # ── Overall comparison ────────────────────────────────────────────
    print("\n── Overall (averaged across all topologies) ──")
    overall = (df.groupby("model")
               .agg(
                   top1_acc=("top1", "mean"),
                   top5_acc=("top5", "mean"),
                   top10_acc=("top10", "mean"),
                   mean_hop_err=("avg_hop_error", "mean"),
                   mean_first_detect=("first_detect_time", "mean"),
                   mean_time_s=("wall_time_s", "mean"),
               )
               .reset_index()
               .sort_values("top1_acc", ascending=False))
    print(overall.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs("evaluation_results", exist_ok=True)
    df.to_csv("evaluation_results/full_results.csv", index=False)
    summary.to_csv("evaluation_results/summary_by_topology.csv", index=False)
    overall.to_csv("evaluation_results/overall_comparison.csv", index=False)

    print("\n✓ All results saved to evaluation_results/")
    return df, summary, overall


# ═══════════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_evaluation(
        topologies=["ER", "BA", "WS", "Regular", "SBM"],
        n_nodes=200,
        n_trials=20,
        train_epochs=150,
        n_train_trees=30,
    )
