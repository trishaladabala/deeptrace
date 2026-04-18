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
* top_k_tracing()       — the new DeepTrace++ forward tracing loop
* adaptive_tracing()    — confidence-guided adaptive expansion (DeepTrace++²)
* bfs_tracing_v2()      — thin wrapper around the original BFS logic,
                           re-implemented here to share the same metric
                           logging interface as the top-k strategy
* dfs_tracing_v2()      — same for DFS
* compare_strategies()  — runs all four on the same graph and returns
                           a comparison DataFrame
* TracingMetrics        — lightweight dataclass that records per-step metrics

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
from confidence import compute_confidence, adaptive_k, ConfidenceTracker
from temporal import TemporalTracingState


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


def _gnn_scores_temporal(
    model,
    subgraph:   nx.Graph,
    state:      TemporalTracingState,
    step:       int,
    node_order: List[int],
) -> Tuple[Dict[int, float], np.ndarray]:
    """
    Temporal-aware GNN inference.

    Extends _gnn_scores() by:
      1. Augmenting base node features with 3 temporal features
         (discovery_order, time_step, growth_stage).
      2. Running GNN forward and extracting hidden embeddings.
      3. Smoothing embeddings via EmbeddingMemory (EMA).
      4. Returning both scores (for source prediction) and smoothed
         embeddings (for drift logging + next-step GRU warm-start).

    Falls back to _mle_scores() + zero embeddings if model is None
    or the subgraph is too small.

    Parameters
    ----------
    model      : TemporalSAGE or SAGE (or None)
    subgraph   : current observed subgraph G_n
    state      : TemporalTracingState carrying feature augmentor + memory
    step       : current tracing step index (1-based)
    node_order : ordered list of valid node IDs in observed set

    Returns
    -------
    (scores dict, smoothed_embeddings ndarray shape (N, hid_dim))
    """
    if model is None or subgraph.number_of_nodes() < 2:
        return _mle_scores(subgraph), np.zeros((subgraph.number_of_nodes(),
                                                state.embed_dim))
    try:
        relabelled  = nx.convert_node_labels_to_integers(
            subgraph, label_attribute="orig")
        orig_label  = nx.get_node_attributes(relabelled, "orig")

        graph_proc  = TreeDataProcess(relabelled)
        unift_list  = graph_proc.get_uninfected_node_list()
        nfeature    = graph_proc.nfeature_process()

        feat_rows   = []
        valid_int   = []          # integer labels in relabelled graph
        valid_orig  = []          # original node IDs
        for k, v in nfeature.items():
            if k not in set(unift_list):
                feat_rows.append([
                    v["node_num"], v["degree_per"], v["degree_per_aver"],
                    v["inft_ndegree_per"], v["inft_alldegree_per"],
                    v["distance_per"], v["layer_rate"], v["layer_num"]
                ])
                valid_int.append(k)
                valid_orig.append(orig_label[k])

        if not feat_rows:
            return _mle_scores(subgraph), np.zeros(
                (subgraph.number_of_nodes(), state.embed_dim))

        base_arr = np.array(feat_rows, dtype=np.float32)

        # ── Append 3 temporal features ─────────────────────────────────
        aug_arr  = state.augment(base_arr, valid_orig)   # (N, 11)
        feats    = th.tensor(aug_arr, dtype=th.float32)

        relabelled.remove_nodes_from(list(unift_list))
        dgl_g    = from_networkx(relabelled)

        model.eval()
        with th.no_grad():
            # Use get_embeddings if model supports it (TemporalSAGE)
            if hasattr(model, 'get_embeddings'):
                # Build prev_hidden from memory for GRU variant
                prev_h = None
                if getattr(model, 'use_gru', False):
                    prev_list = []
                    for nid in valid_orig:
                        mem = state._memory.recall(nid)
                        if mem is not None:
                            prev_list.append(mem)
                        else:
                            prev_list.append(
                                np.zeros(state.embed_dim, dtype=np.float32))
                    prev_h = th.tensor(
                        np.stack(prev_list), dtype=th.float32)
                embeddings_t, logits_t = model.get_embeddings(
                    dgl_g, feats, prev_h)
                raw_emb = embeddings_t.numpy()           # (N, hid_dim)
                logits  = logits_t.squeeze(-1).tolist()
            else:
                # Plain SAGE fallback — no embeddings exposed
                logits  = model(dgl_g, feats).squeeze(-1).tolist()
                raw_emb = np.zeros(
                    (len(valid_orig), state.embed_dim), dtype=np.float32)

        # ── Smooth embeddings via EMA ───────────────────────────────────
        smoothed_emb = state.smooth_embeddings(valid_orig, raw_emb)

        # ── Map back to original labels ─────────────────────────────────
        scores: Dict[int, float] = {}
        for orig, score in zip(valid_orig, logits):
            scores[orig] = score
        return scores, smoothed_emb

    except Exception:
        return _mle_scores(subgraph), np.zeros(
            (subgraph.number_of_nodes(), state.embed_dim))


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
    strategy:          str
    k:                 int            = 1          # only used for top-k
    steps:             int            = 0          # expansion iterations
    nodes_explored:    int            = 0
    correct:           bool           = False
    hop_errors:        List[int]      = field(default_factory=list)
    nodes_at_step:     List[int]      = field(default_factory=list)
    wall_time_s:       float          = 0.0
    # ── Confidence-guided fields (populated by adaptive_tracing) ──────
    confidence_history: List[float]   = field(default_factory=list)
    k_history:          List[int]     = field(default_factory=list)

    def to_dict(self):
        d = {
            "strategy":       self.strategy,
            "k":              self.k,
            "steps":          self.steps,
            "nodes_explored": self.nodes_explored,
            "correct":        self.correct,
            "final_hop_err":  self.hop_errors[-1] if self.hop_errors else None,
            "wall_time_s":    round(self.wall_time_s, 4),
        }
        if self.confidence_history:
            d["mean_confidence"] = round(float(np.mean(self.confidence_history)), 4)
            d["mean_k"]          = round(float(np.mean(self.k_history)), 2)
        return d


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
# 2. Confidence-Guided Adaptive Forward Tracing  (DeepTrace++²)
# ---------------------------------------------------------------------------

def adaptive_tracing(
    G:                 nx.Graph,
    true_source:       int,
    model,
    seed_node:         Optional[int]  = None,
    k_min:             int            = 1,
    k_max:             int            = 10,
    confidence_method: str            = 'margin',
    k_strategy:        str            = 'linear',
    fallback:          str            = 'bfs',       # 'bfs' | 'dfs' | 'none'
    fallback_threshold: float         = 0.1,
    log_every:         int            = 1,
) -> Tuple[TracingMetrics, ConfidenceTracker]:
    """
    Confidence-guided adaptive forward tracing.

    At each step:
      1. Run GNN inference on the current observed subgraph G_n.
      2. Compute a confidence score from the GNN predictions.
      3. Map confidence → dynamic expansion budget k via adaptive_k().
      4. Select the top-k frontier nodes (or fall back to BFS/DFS order
         when confidence is extremely low).
      5. Log all per-step metrics via ConfidenceTracker.

    Parameters
    ----------
    G                  : full epidemic contact graph (networkx)
    true_source        : ground-truth superspreader node id
    model              : trained SAGE model (or None → MLE fallback)
    seed_node          : starting observation node (default: 30th percentile)
    k_min              : expansion budget when confidence is HIGH (exploit)
    k_max              : expansion budget when confidence is LOW  (explore)
    confidence_method  : 'margin' | 'entropy' | 'gap_ratio'
    k_strategy         : 'linear' | 'exponential' | 'step'
    fallback           : what to do when confidence < fallback_threshold
                         'bfs' → BFS-order expansion
                         'dfs' → DFS-order expansion
                         'none' → use GNN scores regardless
    fallback_threshold : confidence below which fallback is triggered
    log_every          : record metrics every `log_every` steps

    Returns
    -------
    (TracingMetrics, ConfidenceTracker)
    """
    nodes     = sorted(G.nodes())
    seed_node = seed_node if seed_node is not None else nodes[int(np.ceil(len(nodes) * 0.3))]

    observed_nodes: set            = {seed_node}
    observed_edges: List[Tuple]    = []

    metrics = TracingMetrics(strategy="adaptive", k=0)
    tracker = ConfidenceTracker()
    t_start = time.time()

    # Pre-compute BFS/DFS order for fallback
    _bfs_order = list(nx.bfs_tree(G, seed_node).nodes())  if fallback == 'bfs' else []
    _dfs_order = list(nx.dfs_tree(G, seed_node).nodes())  if fallback == 'dfs' else []
    _fallback_iter = iter(_bfs_order[1:] if fallback == 'bfs' else
                         _dfs_order[1:] if fallback == 'dfs' else [])

    step = 0
    while len(observed_nodes) < G.number_of_nodes():
        # Build current observed subgraph
        G_n = G.subgraph(sorted(observed_nodes)).copy()

        # ── GNN inference on G_n  ─────────────────────────────────────────
        scores = _gnn_scores(model, G_n)

        # ── Compute confidence  ───────────────────────────────────────────
        conf = compute_confidence(scores, method=confidence_method)

        # ── Derive dynamic k  ─────────────────────────────────────────────
        k = adaptive_k(conf, k_min=k_min, k_max=k_max, strategy=k_strategy)

        # ── Find frontier  ────────────────────────────────────────────────
        frontier: Dict[int, float] = {}
        for obs_node in observed_nodes:
            for nbr in G.neighbors(obs_node):
                if nbr not in observed_nodes:
                    proxy_score = scores.get(obs_node, float("-inf"))
                    if nbr not in frontier or proxy_score > frontier[nbr]:
                        frontier[nbr] = proxy_score

        if not frontier:
            break  # Fully explored

        # ── Select nodes to add  ──────────────────────────────────────────
        use_fallback = (fallback != 'none' and conf < fallback_threshold)

        if use_fallback:
            # Low confidence → revert to systematic BFS/DFS order
            new_nodes = []
            while len(new_nodes) < k:
                try:
                    candidate = next(_fallback_iter)
                    if candidate not in observed_nodes:
                        new_nodes.append(candidate)
                except StopIteration:
                    break
            if not new_nodes:
                # Fallback exhausted, use frontier scores anyway
                new_nodes = sorted(frontier, key=lambda n: frontier[n],
                                   reverse=True)[:k]
        else:
            # Normal: GNN-guided top-k selection
            new_nodes = sorted(frontier, key=lambda n: frontier[n],
                               reverse=True)[:k]

        for new_node in new_nodes:
            observed_nodes.add(new_node)
            for obs in observed_nodes:
                if G.has_edge(new_node, obs) and obs != new_node:
                    observed_edges.append((new_node, obs))

        step += 1
        metrics.steps           = step
        metrics.nodes_explored  = len(observed_nodes)
        metrics.confidence_history.append(conf)
        metrics.k_history.append(k)

        # ── Per-step logging  ─────────────────────────────────────────────
        if step % log_every == 0:
            predicted = _predicted_source(scores)
            hop_err   = _hop_error(G, predicted, true_source)
            metrics.hop_errors.append(hop_err)
            metrics.nodes_at_step.append(len(observed_nodes))

            tracker.record(
                step       = step,
                conf       = conf,
                k          = k,
                n_explored = len(observed_nodes),
                hop_err    = hop_err,
                pred_src   = predicted,
            )

    # ── Final evaluation on the full graph  ────────────────────────────────
    full_scores       = _gnn_scores(model, G)
    final_pred        = _predicted_source(full_scores)
    metrics.correct   = (final_pred == true_source)
    metrics.wall_time_s = time.time() - t_start

    if not metrics.hop_errors:
        metrics.hop_errors.append(_hop_error(G, final_pred, true_source))
        metrics.nodes_at_step.append(len(observed_nodes))

    conv_step = tracker.convergence_step
    print(f"[Adaptive k∈[{k_min},{k_max}] {confidence_method}/{k_strategy}] "
          f"steps={metrics.steps} | "
          f"nodes_explored={metrics.nodes_explored} | "
          f"correct={metrics.correct} | "
          f"final_hop_err={metrics.hop_errors[-1]} | "
          f"mean_conf={np.mean(metrics.confidence_history):.3f} | "
          f"mean_k={np.mean(metrics.k_history):.1f} | "
          f"converged@step={conv_step} | "
          f"time={metrics.wall_time_s:.3f}s")

    return metrics, tracker


# ---------------------------------------------------------------------------
# 3. Temporal-Aware Adaptive Tracing  (DeepTrace++ temporal)
# ---------------------------------------------------------------------------

def temporal_adaptive_tracing(
    G:                 nx.Graph,
    true_source:       int,
    model,
    seed_node:         Optional[int] = None,
    k_min:             int           = 1,
    k_max:             int           = 10,
    confidence_method: str           = 'margin',
    k_strategy:        str           = 'linear',
    fallback:          str           = 'bfs',
    fallback_threshold: float        = 0.1,
    embed_dim:         int           = 50,
    alpha:             float         = 0.3,
    use_gru:           bool          = False,
    log_every:         int           = 1,
) -> Tuple[TracingMetrics, ConfidenceTracker, TemporalTracingState]:
    """
    Temporal-aware adaptive tracing — combines:
      * Temporal feature augmentation (discovery_order, time_step, growth_stage)
      * Embedding memory with EMA smoothing across steps
      * Optional GRU recurrent update (if model is TemporalSAGE with use_gru)
      * Confidence-guided adaptive expansion budget k
      * BFS/DFS fallback for low-confidence early steps

    Parameters
    ----------
    G, true_source, model, seed_node : same as adaptive_tracing()
    k_min, k_max, confidence_method,
    k_strategy, fallback,
    fallback_threshold               : same as adaptive_tracing()
    embed_dim          : hidden embedding dimension (must match model hid_feats)
    alpha              : EMA smoothing factor ∈ (0, 1]
    use_gru            : whether the model uses GRU recurrence
    log_every          : metric logging frequency

    Returns
    -------
    (TracingMetrics, ConfidenceTracker, TemporalTracingState)
    """
    nodes     = sorted(G.nodes())
    seed_node = (seed_node if seed_node is not None
                 else nodes[int(np.ceil(len(nodes) * 0.3))])

    observed_nodes: set         = {seed_node}
    observed_edges: List[Tuple] = []

    metrics = TracingMetrics(strategy="temporal_adaptive", k=0)
    tracker = ConfidenceTracker()
    t_state = TemporalTracingState(
        total_nodes=G.number_of_nodes(),
        embed_dim=embed_dim,
        alpha=alpha,
        use_gru=use_gru,
    )
    t_state.register_seed(seed_node)
    t_start = time.time()

    # Pre-compute BFS/DFS order for fallback
    _bfs_order    = list(nx.bfs_tree(G, seed_node).nodes()) \
                    if fallback == 'bfs' else []
    _dfs_order    = list(nx.dfs_tree(G, seed_node).nodes()) \
                    if fallback == 'dfs' else []
    _fallback_iter = iter(_bfs_order[1:] if fallback == 'bfs' else
                          _dfs_order[1:] if fallback == 'dfs' else [])

    step = 0
    while len(observed_nodes) < G.number_of_nodes():
        G_n = G.subgraph(sorted(observed_nodes)).copy()

        # ── Temporal-aware GNN inference ──────────────────────────────
        scores, _smoothed_emb = _gnn_scores_temporal(
            model, G_n, t_state,
            step=step,
            node_order=sorted(observed_nodes),
        )

        # ── Confidence → dynamic k ────────────────────────────────────
        conf = compute_confidence(scores, method=confidence_method)
        k    = adaptive_k(conf, k_min=k_min, k_max=k_max,
                          strategy=k_strategy)

        # ── Find frontier ─────────────────────────────────────────────
        frontier: Dict[int, float] = {}
        for obs_node in observed_nodes:
            for nbr in G.neighbors(obs_node):
                if nbr not in observed_nodes:
                    proxy = scores.get(obs_node, float("-inf"))
                    if nbr not in frontier or proxy > frontier[nbr]:
                        frontier[nbr] = proxy

        if not frontier:
            break

        # ── Select nodes ──────────────────────────────────────────────
        use_fallback = (fallback != 'none' and conf < fallback_threshold)
        if use_fallback:
            new_nodes = []
            while len(new_nodes) < k:
                try:
                    cand = next(_fallback_iter)
                    if cand not in observed_nodes:
                        new_nodes.append(cand)
                except StopIteration:
                    break
            if not new_nodes:
                new_nodes = sorted(frontier, key=lambda n: frontier[n],
                                   reverse=True)[:k]
        else:
            new_nodes = sorted(frontier, key=lambda n: frontier[n],
                               reverse=True)[:k]

        step += 1

        # ── Register temporal state for new nodes ──────────────────────
        t_state.register_new_nodes(
            new_nodes, step=step,
            current_graph_size=len(observed_nodes) + len(new_nodes))

        for new_node in new_nodes:
            observed_nodes.add(new_node)
            for obs in observed_nodes:
                if G.has_edge(new_node, obs) and obs != new_node:
                    observed_edges.append((new_node, obs))

        metrics.steps          = step
        metrics.nodes_explored = len(observed_nodes)
        metrics.confidence_history.append(conf)
        metrics.k_history.append(k)

        # ── Log per-step metrics ──────────────────────────────────────
        if step % log_every == 0:
            predicted = _predicted_source(scores)
            hop_err   = _hop_error(G, predicted, true_source)
            metrics.hop_errors.append(hop_err)
            metrics.nodes_at_step.append(len(observed_nodes))
            t_state.log_prediction(predicted)

            tracker.record(
                step=step, conf=conf, k=k,
                n_explored=len(observed_nodes),
                hop_err=hop_err, pred_src=predicted,
            )

    # ── Final evaluation ──────────────────────────────────────────────
    full_scores, _ = _gnn_scores_temporal(
        model, G, t_state,
        step=step, node_order=sorted(G.nodes()),
    )
    final_pred        = _predicted_source(full_scores)
    metrics.correct   = (final_pred == true_source)
    metrics.wall_time_s = time.time() - t_start

    if not metrics.hop_errors:
        metrics.hop_errors.append(_hop_error(G, final_pred, true_source))
        metrics.nodes_at_step.append(len(observed_nodes))

    conv_step = tracker.convergence_step
    t_summary = t_state.summary()
    print(
        f"[Temporal k∈[{k_min},{k_max}] {confidence_method}/{k_strategy} "
        f"alpha={alpha} gru={use_gru}] "
        f"steps={metrics.steps} | "
        f"nodes_explored={metrics.nodes_explored} | "
        f"correct={metrics.correct} | "
        f"final_hop_err={metrics.hop_errors[-1]} | "
        f"mean_conf={np.mean(metrics.confidence_history):.3f} | "
        f"stability={t_summary['prediction_stability']:.3f} | "
        f"drift={t_summary['mean_embedding_drift']} | "
        f"converged@step={conv_step} | "
        f"time={metrics.wall_time_s:.3f}s"
    )
    return metrics, tracker, t_state


# ---------------------------------------------------------------------------
# 3. BFS Forward Tracing (comparison baseline)
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
# 4. DFS Forward Tracing (comparison baseline)
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
# 5. Comparison runner
# ---------------------------------------------------------------------------

def compare_strategies(
    G:                   nx.Graph,
    true_source:         int,
    model                = None,
    k_values:            List[int]           = (1, 3, 5),
    seed_node:           Optional[int]       = None,
    log_every:           int                 = 1,
    adaptive_configs:    Optional[List[dict]] = None,
    temporal_configs:    Optional[List[dict]] = None,
) -> Tuple[pd.DataFrame, Dict[str, ConfidenceTracker],
           Dict[str, TemporalTracingState]]:
    """
    Run BFS, DFS, Top-K, Adaptive, and Temporal Adaptive on the same
    graph and return a summary DataFrame with metrics for every strategy.

    Parameters
    ----------
    G, true_source, model, k_values,
    seed_node, log_every, adaptive_configs : same as before
    temporal_configs : list of dicts with kwargs for temporal_adaptive_tracing.
                       Set to [] to skip temporal runs.
                       Defaults to one run with default parameters.

    Returns
    -------
    (DataFrame, confidence_trackers dict, temporal_states dict)
    """
    if adaptive_configs is None:
        adaptive_configs = [
            {"confidence_method": "margin", "k_strategy": "linear"},
        ]
    if temporal_configs is None:
        temporal_configs = [
            {"confidence_method": "margin", "k_strategy": "linear",
             "alpha": 0.3, "use_gru": False},
        ]

    results   = []
    trackers:  Dict[str, ConfidenceTracker]  = {}
    t_states:  Dict[str, TemporalTracingState] = {}

    print("=" * 65)
    print(f"Graph: {G.number_of_nodes()} nodes | true_source={true_source}")
    print("=" * 65)

    results.append(
        bfs_tracing_v2(G, true_source, model, seed_node, log_every).to_dict())
    results.append(
        dfs_tracing_v2(G, true_source, model, seed_node, log_every).to_dict())

    for k in k_values:
        m = top_k_tracing(G, true_source, model, seed_node,
                          k=k, log_every=log_every)
        results.append(m.to_dict())

    for cfg in adaptive_configs:
        label = (f"adaptive_"
                 f"{cfg.get('confidence_method','margin')}_"
                 f"{cfg.get('k_strategy','linear')}")
        m, tracker = adaptive_tracing(
            G, true_source, model, seed_node,
            log_every=log_every, **cfg)
        results.append(m.to_dict())
        trackers[label] = tracker

    for cfg in temporal_configs:
        label = (f"temporal_"
                 f"{cfg.get('confidence_method','margin')}_"
                 f"{cfg.get('k_strategy','linear')}_"
                 f"alpha{cfg.get('alpha', 0.3)}_"
                 f"gru{cfg.get('use_gru', False)}")
        m, tracker, t_state = temporal_adaptive_tracing(
            G, true_source, model, seed_node,
            log_every=log_every, **cfg)
        results.append(m.to_dict())
        trackers[label] = tracker
        t_states[label] = t_state

    df = pd.DataFrame(results)
    print("\n── Strategy Comparison ─────────────────────────────────")
    print(df.to_string(index=False))
    return df, trackers, t_states


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import random
    from math import floor
    from model import SAGE, TemporalSAGE, train_data_process
    import dgl

    random.seed(42)
    np.random.seed(42)
    th.manual_seed(42)

    # ── Build a small test graph ───────────────────────────
    node_num    = 50
    degree      = floor(node_num * 3 / 20)
    ER          = nx.random_graphs.watts_strogatz_graph(node_num, degree, 0.3)
    G           = nx.minimum_spanning_tree(ER, algorithm="kruskal")
    G           = nx.convert_node_labels_to_integers(G)
    true_source = 0

    # ── Train static SAGE (8 features) ───────────────────────
    print("Training static SAGE (8 feat, 10 epochs)…")
    all_trees   = [train_data_process(2, 60 + i) for i in range(10)]
    train_batch = dgl.batch(all_trees)
    feat_dim    = train_batch.ndata["feat"].shape[1]   # 8
    model_static = SAGE(in_feats=feat_dim, hid_feats=50, out_feats=1)
    opt = th.optim.Adam(model_static.parameters())
    for epoch in range(10):
        model_static.train()
        logits = model_static(train_batch, train_batch.ndata["feat"])
        loss   = F.mse_loss(logits.squeeze(-1), train_batch.ndata["labels"])
        opt.zero_grad(); loss.backward(); opt.step()
    print("Static model ready.\n")

    # ── Train TemporalSAGE smoothing-only (11 features) ────────
    # Note: temporal features are injected during tracing, not training.
    # We train on the base 8-feature data and let the extra dims be zero
    # at batch training time (they carry no signal without a real run).
    # A production system would generate training data from simulated runs.
    print("Training TemporalSAGE-smooth (11 feat, 10 epochs)…")
    model_temp_smooth = TemporalSAGE(
        in_feats=feat_dim + 3,   # 11
        hid_feats=50, out_feats=1, use_gru=False)
    opt2 = th.optim.Adam(model_temp_smooth.parameters())
    # Pad training features with zeros for temporal dims
    train_feats_padded = th.cat([
        train_batch.ndata["feat"],
        th.zeros(train_batch.ndata["feat"].shape[0], 3)
    ], dim=1)
    for epoch in range(10):
        model_temp_smooth.train()
        logits = model_temp_smooth(train_batch, train_feats_padded)
        loss   = F.mse_loss(logits.squeeze(-1), train_batch.ndata["labels"])
        opt2.zero_grad(); loss.backward(); opt2.step()
    print("Temporal-smooth model ready.\n")

    # ── Train TemporalSAGE + GRU (11 features + recurrence) ───
    print("Training TemporalSAGE-GRU (11 feat + GRU, 10 epochs)…")
    model_temp_gru = TemporalSAGE(
        in_feats=feat_dim + 3,
        hid_feats=50, out_feats=1, use_gru=True)
    opt3 = th.optim.Adam(model_temp_gru.parameters())
    for epoch in range(10):
        model_temp_gru.train()
        logits = model_temp_gru(train_batch, train_feats_padded)
        loss   = F.mse_loss(logits.squeeze(-1), train_batch.ndata["labels"])
        opt3.zero_grad(); loss.backward(); opt3.step()
    print("Temporal-GRU model ready.\n")

    # ── Adaptive configs (shared base) ─────────────────────
    adaptive_configs = [
        {"confidence_method": "margin", "k_strategy": "linear",
         "k_min": 1, "k_max": 8},
    ]

    # ── Temporal configs (both variants) ──────────────────
    temporal_configs = [
        {"confidence_method": "margin", "k_strategy": "linear",
         "k_min": 1, "k_max": 8, "alpha": 0.3, "use_gru": False,
         "embed_dim": 50},
        {"confidence_method": "margin", "k_strategy": "linear",
         "k_min": 1, "k_max": 8, "alpha": 0.3, "use_gru": True,
         "embed_dim": 50},
    ]

    # ── Run full comparison ──────────────────────────────
    # Note: temporal runs use model_temp_smooth / model_temp_gru
    # but compare_strategies passes a single model; run separately.
    print("\n--- Static model comparison ---")
    df_static, trackers_s, _ = compare_strategies(
        G, true_source, model_static,
        k_values=[1, 3, 5],
        adaptive_configs=adaptive_configs,
        temporal_configs=[],           # skip temporal in this pass
    )

    print("\n--- Temporal-smooth model comparison ---")
    df_temp_s, trackers_ts, t_states_s = compare_strategies(
        G, true_source, model_temp_smooth,
        k_values=[],                   # skip plain top-k
        adaptive_configs=[],
        temporal_configs=[
            {"confidence_method": "margin", "k_strategy": "linear",
             "k_min": 1, "k_max": 8, "alpha": 0.3, "use_gru": False,
             "embed_dim": 50},
        ],
    )

    print("\n--- Temporal-GRU model comparison ---")
    df_temp_g, trackers_tg, t_states_g = compare_strategies(
        G, true_source, model_temp_gru,
        k_values=[],
        adaptive_configs=[],
        temporal_configs=[
            {"confidence_method": "margin", "k_strategy": "linear",
             "k_min": 1, "k_max": 8, "alpha": 0.3, "use_gru": True,
             "embed_dim": 50},
        ],
    )

    # ── Combine and save ──────────────────────────────────
    log_dir = "adaptive_tracing_logs"
    os.makedirs(log_dir, exist_ok=True)

    df_all = pd.concat([df_static, df_temp_s, df_temp_g],
                       ignore_index=True)
    df_all.to_csv(os.path.join(log_dir, "comparison_results.csv"),
                  index=False)
    print(f"\nAll comparison results → {log_dir}/comparison_results.csv")

    # Temporal-specific logs
    all_t_states = {**t_states_s, **t_states_g}
    temporal_rows = []
    for label, ts in all_t_states.items():
        row = {"label": label, **ts.summary()}
        temporal_rows.append(row)
        # Per-step confidence trace
        trkr = {**trackers_ts, **trackers_tg}.get(label)
        if trkr:
            trace_df = trkr.to_dataframe()
            # Merge embedding drift into trace
            drift_col = ts.embedding_drifts
            if len(drift_col) >= len(trace_df):
                trace_df["embedding_drift"] = (
                    drift_col[:len(trace_df)])
            trace_path = os.path.join(
                log_dir, f"temporal_trace_{label}.csv")
            trace_df.to_csv(trace_path, index=False)
            print(f"Temporal trace → {trace_path}")

    if temporal_rows:
        pd.DataFrame(temporal_rows).to_csv(
            os.path.join(log_dir, "temporal_comparison.csv"), index=False)
        print(f"Temporal summary → {log_dir}/temporal_comparison.csv")

    # Backward-compatible
    df_static.to_csv("topk_vs_bfs_dfs.csv", index=False)
    print("Static comparison → topk_vs_bfs_dfs.csv")

