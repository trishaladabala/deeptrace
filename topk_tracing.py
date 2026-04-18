"""
topk_tracing.py  —  DeepTrace++ Adaptive Top-K Forward Tracing
================================================================
Implements the adaptive top-k forward tracing strategy described in
DeepTrace++. Instead of expanding the observed graph G_n via static
BFS/DFS order, we run the trained GNN at every step, score each
frontier node with its predicted likelihood score  y_hat_v ≈ log P(G_n | v),
and expand only the top-k highest-scoring frontier nodes.

Key components
--------------
* top_k_tracing()   — the new DeepTrace++ forward tracing loop
* bfs_tracing_v2()  — thin wrapper around the original BFS logic,
                       re-implemented here to share the same metric
                       logging interface as the top-k strategy
* dfs_tracing_v2()  — same for DFS
* compare_strategies()  — runs all three on the same graph and returns
                          a comparison DataFrame
* TracingMetrics    — lightweight dataclass that records per-step metrics

Usage
-----
    from topk_tracing import compare_strategies
    from model import SAGE, train_data_process
    import dgl, torch as th

    # Load a pre-trained model (or pass None to use the MLE heuristic)
    model = ...
    results_df = compare_strategies(G, true_source, model, k=3)
    print(results_df)
"""

import collections
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
from dgl.convert import from_networkx

from cal_max_min_ds import CalMaxMinDS
from graph_data_process import TreeDataProcess


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _gnn_scores(model, subgraph: nx.Graph) -> Dict[int, float]:
    """
    Run the GNN on *subgraph* and return a dict  {node_id: score}.
    Score = the raw scalar output of the last GraphSAGE layer, which
    approximates log P(G_n | v).  Higher is "more likely superspreader".

    If the graph is too small (< 2 nodes) or the model is None the
    function falls back to the MLE geometric-probability heuristic so
    that the top-k loop degrades gracefully.
    """
    if model is None or subgraph.number_of_nodes() < 2:
        return _mle_scores(subgraph)

    try:
        # Re-label to 0-indexed integers (names may be arbitrary strings)
        relabelled = nx.convert_node_labels_to_integers(subgraph, label_attribute="orig")
        orig_label = nx.get_node_attributes(relabelled, "orig")

        graph_proc = TreeDataProcess(relabelled)
        unift_list  = graph_proc.get_uninfected_node_list()
        nfeature    = graph_proc.nfeature_process()

        feat_rows = []
        valid_nodes = []
        for k, v in nfeature.items():
            if k not in set(unift_list):
                feat_rows.append([
                    v["node_num"], v["degree_per"], v["degree_per_aver"],
                    v["inft_ndegree_per"], v["inft_alldegree_per"],
                    v["distance_per"], v["layer_rate"], v["layer_num"]
                ])
                valid_nodes.append(k)

        if not feat_rows:
            return _mle_scores(subgraph)

        feats  = th.tensor(feat_rows, dtype=th.float32)
        relabelled.remove_nodes_from(list(unift_list))
        dgl_g  = from_networkx(relabelled)

        model.eval()
        with th.no_grad():
            logits = model(dgl_g, feats).squeeze(-1).tolist()

        # Map back to original node labels
        scores = {}
        for idx, score in zip(valid_nodes, logits):
            original = orig_label[idx]
            scores[original] = score
        return scores

    except Exception:
        return _mle_scores(subgraph)


def _mle_scores(subgraph: nx.Graph) -> Dict[int, float]:
    """
    Fallback: geometric mean of max/min DS probabilities (original MLE heuristic).
    """
    degree_list     = nx.degree(subgraph)
    leaf_list       = [v for v, d in degree_list if d == 1]
    non_leaf_nodes  = [v for v in subgraph.nodes() if v not in leaf_list]
    scores: Dict[int, float] = {}
    for node in non_leaf_nodes:
        try:
            cal = CalMaxMinDS(subgraph, leaf_list, node)
            mx  = cal.cal_max_ds()
            mn  = cal.cal_min_ds()
            scores[node] = math.sqrt(mx * mn)
        except Exception:
            scores[node] = float("-inf")
    # Leaves get the minimum score
    min_score = min(scores.values(), default=float("-inf"))
    for leaf in leaf_list:
        scores[leaf] = min_score
    return scores


def _predicted_source(scores: Dict[int, float]) -> int:
    """Return the node with the highest predicted score."""
    return max(scores, key=lambda n: scores[n])


def _hop_error(G: nx.Graph, predicted: int, true_source: int) -> int:
    """Shortest-path hop distance between predicted and true source in G."""
    try:
        return len(nx.shortest_path(G, predicted, true_source)) - 1
    except nx.NetworkXNoPath:
        return -1


# ---------------------------------------------------------------------------
# Metric container
# ---------------------------------------------------------------------------

@dataclass
class TracingMetrics:
    strategy:        str
    k:               int            = 1          # only used for top-k
    steps:           int            = 0          # expansion iterations
    nodes_explored:  int            = 0
    correct:         bool           = False
    hop_errors:      List[int]      = field(default_factory=list)
    nodes_at_step:   List[int]      = field(default_factory=list)
    wall_time_s:     float          = 0.0

    def to_dict(self):
        return {
            "strategy":       self.strategy,
            "k":              self.k,
            "steps":          self.steps,
            "nodes_explored": self.nodes_explored,
            "correct":        self.correct,
            "final_hop_err":  self.hop_errors[-1] if self.hop_errors else None,
            "wall_time_s":    round(self.wall_time_s, 4),
        }


# ---------------------------------------------------------------------------
# 1. Top-K Adaptive Forward Tracing  (DeepTrace++)
# ---------------------------------------------------------------------------

def top_k_tracing(
    G:           nx.Graph,
    true_source: int,
    model,
    seed_node:   Optional[int] = None,
    k:           int           = 3,
    log_every:   int           = 1,
) -> TracingMetrics:
    """
    Adaptive Top-K forward tracing (DeepTrace++).

    At each step:
      1. Compute GNN likelihood scores for all nodes in the current
         observed subgraph  G_n.
      2. Identify the frontier  F = neighbours(G_n) that are not yet
         in G_n.
      3. Score each frontier node using the predicted likelihood of its
         already-observed neighbour (proxy for how "attractive" the
         direction is).
      4. Add the top-k frontier nodes to G_n.
      5. Log detection accuracy and hop error.

    Parameters
    ----------
    G           : full epidemic contact graph (networkx)
    true_source : ground-truth superspreader node id
    model       : trained SAGE model (or None → MLE fallback)
    seed_node   : starting observation node (default: node at 30th percentile)
    k           : number of frontier nodes added per step
    log_every   : record metrics every `log_every` steps

    Returns
    -------
    TracingMetrics
    """
    nodes      = sorted(G.nodes())
    seed_node  = seed_node if seed_node is not None else nodes[int(np.ceil(len(nodes) * 0.3))]

    observed_nodes: set = {seed_node}
    observed_edges: List[Tuple] = []

    metrics = TracingMetrics(strategy="top_k", k=k)
    t_start = time.time()

    step = 0
    while len(observed_nodes) < G.number_of_nodes():
        # Build current observed subgraph
        G_n = G.subgraph(sorted(observed_nodes)).copy()

        # ── Score all nodes in G_n with the GNN/MLE  ──────────────────────
        scores = _gnn_scores(model, G_n)

        # ── Find frontier: neighbours of G_n not yet observed  ────────────
        frontier: Dict[int, float] = {}
        for obs_node in observed_nodes:
            for nbr in G.neighbors(obs_node):
                if nbr not in observed_nodes:
                    # Frontier score = score of the observed neighbour
                    # (we want to expand from high-scoring seeds first)
                    proxy_score = scores.get(obs_node, float("-inf"))
                    # Keep the best observed-neighbour score per frontier node
                    if nbr not in frontier or proxy_score > frontier[nbr]:
                        frontier[nbr] = proxy_score

        if not frontier:
            break  # Fully explored

        # ── Select top-k frontier nodes  ──────────────────────────────────
        top_k_nodes = sorted(frontier, key=lambda n: frontier[n], reverse=True)[:k]

        for new_node in top_k_nodes:
            observed_nodes.add(new_node)
            # Add the connecting edge(s) to observed_edges
            for obs in observed_nodes:
                if G.has_edge(new_node, obs) and obs != new_node:
                    observed_edges.append((new_node, obs))

        step += 1
        metrics.steps          = step
        metrics.nodes_explored = len(observed_nodes)

        if step % log_every == 0:
            predicted = _predicted_source(scores)
            hop_err   = _hop_error(G, predicted, true_source)
            metrics.hop_errors.append(hop_err)
            metrics.nodes_at_step.append(len(observed_nodes))

    # Final evaluation on the full graph
    full_scores    = _gnn_scores(model, G)
    final_pred     = _predicted_source(full_scores)
    metrics.correct      = (final_pred == true_source)
    metrics.wall_time_s  = time.time() - t_start

    if not metrics.hop_errors:
        metrics.hop_errors.append(_hop_error(G, final_pred, true_source))
        metrics.nodes_at_step.append(len(observed_nodes))

    print(f"[Top-K k={k}] steps={metrics.steps} | "
          f"nodes_explored={metrics.nodes_explored} | "
          f"correct={metrics.correct} | "
          f"final_hop_err={metrics.hop_errors[-1]} | "
          f"time={metrics.wall_time_s:.3f}s")

    return metrics


# ---------------------------------------------------------------------------
# 2. BFS Forward Tracing (comparison baseline)
# ---------------------------------------------------------------------------

def bfs_tracing_v2(
    G:           nx.Graph,
    true_source: int,
    model,
    seed_node:   Optional[int] = None,
    log_every:   int           = 10,
) -> TracingMetrics:
    """BFS expansion — identical behaviour to original bfs_tracing but
    uses the same metric-logging interface as top_k_tracing."""
    nodes         = sorted(G.nodes())
    seed_node     = seed_node if seed_node is not None else nodes[int(np.ceil(len(nodes) * 0.3))]

    traced        = {seed_node}
    queue         = collections.deque([seed_node])
    tracing_G     = nx.Graph()

    metrics       = TracingMetrics(strategy="bfs", k=0)
    t_start       = time.time()
    k_edges       = 0

    while queue:
        node      = queue.popleft()
        neighbors = [n for n in G.neighbors(node) if n not in traced]
        for nbr in neighbors:
            traced.add(nbr)
            queue.append(nbr)
            tracing_G.add_edge(node, nbr)
            k_edges += 1
            if k_edges % log_every == 1:
                scores    = _mle_scores(tracing_G)
                predicted = _predicted_source(scores) if scores else node
                hop_err   = _hop_error(G, predicted, true_source)
                metrics.hop_errors.append(hop_err)
                metrics.nodes_at_step.append(len(traced))

    metrics.steps          = k_edges
    metrics.nodes_explored = len(traced)

    full_scores        = _gnn_scores(model, G) if model else _mle_scores(G)
    final_pred         = _predicted_source(full_scores)
    metrics.correct    = (final_pred == true_source)
    metrics.wall_time_s = time.time() - t_start

    if not metrics.hop_errors:
        metrics.hop_errors.append(_hop_error(G, final_pred, true_source))
        metrics.nodes_at_step.append(len(traced))

    print(f"[BFS] steps={metrics.steps} | "
          f"nodes_explored={metrics.nodes_explored} | "
          f"correct={metrics.correct} | "
          f"final_hop_err={metrics.hop_errors[-1]} | "
          f"time={metrics.wall_time_s:.3f}s")
    return metrics


# ---------------------------------------------------------------------------
# 3. DFS Forward Tracing (comparison baseline)
# ---------------------------------------------------------------------------

def dfs_tracing_v2(
    G:           nx.Graph,
    true_source: int,
    model,
    seed_node:   Optional[int] = None,
    log_every:   int           = 10,
) -> TracingMetrics:
    """DFS expansion — mirrors original dfs_tracing logic with shared metrics."""
    nodes         = sorted(G.nodes())
    seed_node     = seed_node if seed_node is not None else nodes[int(np.ceil(len(nodes) * 0.3))]

    traced        = [seed_node]
    traced_set    = {seed_node}
    tracing_G     = nx.Graph()
    furcation     = [seed_node]

    metrics       = TracingMetrics(strategy="dfs", k=0)
    t_start       = time.time()
    k_edges       = 0

    untraced = set(G.nodes()) - traced_set

    while untraced:
        to_trace  = traced[-1]
        neighbors = [n for n in G.neighbors(to_trace) if n not in traced_set]

        if neighbors:
            next_node = neighbors[-1]
            if len(neighbors) > 1:
                furcation.append(to_trace)
        else:
            next_node = None
            while furcation:
                cand_nbrs = [n for n in G.neighbors(furcation[-1]) if n not in traced_set]
                if cand_nbrs:
                    to_trace  = furcation[-1]
                    next_node = cand_nbrs[-1]
                    if len(cand_nbrs) > 1:
                        furcation.append(to_trace)
                    break
                else:
                    furcation.pop()
            if next_node is None:
                break

        untraced.discard(next_node)
        traced.append(next_node)
        traced_set.add(next_node)
        tracing_G.add_edge(to_trace, next_node)
        k_edges += 1

        if k_edges % log_every == 1:
            scores    = _mle_scores(tracing_G)
            predicted = _predicted_source(scores) if scores else to_trace
            hop_err   = _hop_error(G, predicted, true_source)
            metrics.hop_errors.append(hop_err)
            metrics.nodes_at_step.append(len(traced_set))

    metrics.steps          = k_edges
    metrics.nodes_explored = len(traced_set)

    full_scores        = _gnn_scores(model, G) if model else _mle_scores(G)
    final_pred         = _predicted_source(full_scores)
    metrics.correct    = (final_pred == true_source)
    metrics.wall_time_s = time.time() - t_start

    if not metrics.hop_errors:
        metrics.hop_errors.append(_hop_error(G, final_pred, true_source))
        metrics.nodes_at_step.append(len(traced_set))

    print(f"[DFS] steps={metrics.steps} | "
          f"nodes_explored={metrics.nodes_explored} | "
          f"correct={metrics.correct} | "
          f"final_hop_err={metrics.hop_errors[-1]} | "
          f"time={metrics.wall_time_s:.3f}s")
    return metrics


# ---------------------------------------------------------------------------
# 4. Comparison runner
# ---------------------------------------------------------------------------

def compare_strategies(
    G:           nx.Graph,
    true_source: int,
    model       = None,
    k_values:    List[int] = (1, 3, 5),
    seed_node:   Optional[int] = None,
    log_every:   int           = 1,
) -> pd.DataFrame:
    """
    Run BFS, DFS, and Top-K (for each value in k_values) on the same
    graph and return a summary DataFrame with metrics for every strategy.

    Parameters
    ----------
    G           : full contact graph
    true_source : ground-truth superspreader
    model       : trained SAGE (or None)
    k_values    : list of k values to benchmark Top-K with
    seed_node   : common starting node (auto-selected if None)
    log_every   : metric logging frequency

    Returns
    -------
    pd.DataFrame with columns:
        strategy | k | steps | nodes_explored | correct | final_hop_err | wall_time_s
    """
    results = []

    print("=" * 60)
    print(f"Graph: {G.number_of_nodes()} nodes | true_source={true_source}")
    print("=" * 60)

    results.append(bfs_tracing_v2(G, true_source, model, seed_node, log_every).to_dict())
    results.append(dfs_tracing_v2(G, true_source, model, seed_node, log_every).to_dict())

    for k in k_values:
        m = top_k_tracing(G, true_source, model, seed_node, k=k, log_every=log_every)
        results.append(m.to_dict())

    df = pd.DataFrame(results)
    print("\n── Strategy Comparison ─────────────────────────────────")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    from math import floor
    from model import SAGE, train_data_process
    import dgl

    random.seed(42)
    np.random.seed(42)
    th.manual_seed(42)

    # ── Build a small test graph ──────────────────────────────────────────
    node_num   = 50
    degree     = floor(node_num * 3 / 20)
    ER         = nx.random_graphs.watts_strogatz_graph(node_num, degree, 0.3)
    G          = nx.minimum_spanning_tree(ER, algorithm="kruskal")
    G          = nx.convert_node_labels_to_integers(G)
    true_source = 0          # treat node 0 as the known superspreader

    # ── Train a tiny model ───────────────────────────────────────────────
    print("Training GNN (10 epochs for demo)…")
    all_trees   = [train_data_process(2, 60 + i) for i in range(10)]
    train_batch = dgl.batch(all_trees)

    feat_dim = train_batch.ndata["feat"].shape[1]
    model    = SAGE(in_feats=feat_dim, hid_feats=50, out_feats=1)
    opt      = th.optim.Adam(model.parameters())

    for epoch in range(10):
        model.train()
        logits = model(train_batch, train_batch.ndata["feat"])
        loss   = F.mse_loss(logits.squeeze(-1), train_batch.ndata["labels"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("Training done.\n")

    # ── Run comparison ───────────────────────────────────────────────────
    df = compare_strategies(G, true_source, model, k_values=[1, 3, 5])
    df.to_csv("topk_vs_bfs_dfs.csv", index=False)
    print("\nFull results saved to topk_vs_bfs_dfs.csv")
