"""
temporal.py — Temporal (Stage-Wise) Awareness for DeepTrace++
==============================================================
Provides temporal feature augmentation, embedding memory with
exponential smoothing, and a GRU-based recurrent update for
incorporating historical information across tracing steps.

Key API
-------
* TemporalFeatureAugmentor — computes 3 temporal features per node
* EmbeddingMemory          — exponential smoothing across time steps
* GRUTemporalUpdate        — learnable GRU cell for recurrent embeddings
* TemporalTracingState     — orchestrator tying the above together

The module is decoupled from the GNN architecture and tracing loop
so that it can be used as a drop-in augmentation layer.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Temporal Feature Augmentor
# ═══════════════════════════════════════════════════════════════════════════

class TemporalFeatureAugmentor:
    """
    Computes three temporal features for each node in the observed
    subgraph G_n:

        discovery_order  — normalised rank in which the node was added
                           (0.0 = seed node, 1.0 = last discovered)
        time_step        — normalised tracing step t/T_est when the
                           node entered G_n
        growth_stage     — |G_n| / |G_total| at the moment the node
                           was discovered (how "full" the subgraph was)

    All three features are in [0, 1], making them safe to concatenate
    with the existing 8-feature vector without rescaling.
    """

    def __init__(self, total_nodes: int):
        """
        Parameters
        ----------
        total_nodes : total number of nodes in the full graph G
                      (used to normalise growth_stage and time_step)
        """
        self.total_nodes = max(total_nodes, 1)

        # discovery_order is the insertion rank; we store it per node
        self._discovery: OrderedDict[int, int] = OrderedDict()
        # step and graph-size at the moment each node was discovered
        self._step_discovered: Dict[int, int]  = {}
        self._size_at_discovery: Dict[int, int] = {}
        self._current_step: int = 0
        self._max_step_estimate: int = total_nodes  # rough upper bound

    # ── Registration ──────────────────────────────────────────────────

    def register_nodes(self, new_nodes: List[int], step: int,
                       current_graph_size: int) -> None:
        """
        Register newly discovered nodes at the given tracing step.
        Must be called *every step* as nodes are added to G_n.
        """
        self._current_step = step
        for node in new_nodes:
            if node not in self._discovery:
                rank = len(self._discovery)
                self._discovery[node] = rank
                self._step_discovered[node] = step
                self._size_at_discovery[node] = current_graph_size

    def register_seed(self, seed_node: int) -> None:
        """Register the seed node at step 0 with graph size 1."""
        self.register_nodes([seed_node], step=0, current_graph_size=1)

    # ── Feature computation ───────────────────────────────────────────

    def get_temporal_features(self, node_ids: List[int]) -> np.ndarray:
        """
        Return a (len(node_ids), 3) array of temporal features.

        Columns:
            0: discovery_order  ∈ [0, 1]
            1: time_step        ∈ [0, 1]
            2: growth_stage     ∈ [0, 1]
        """
        n = len(node_ids)
        feats = np.zeros((n, 3), dtype=np.float32)

        max_rank = max(len(self._discovery) - 1, 1)
        max_step = max(self._current_step, 1)

        for i, nid in enumerate(node_ids):
            rank = self._discovery.get(nid, max_rank)
            step = self._step_discovered.get(nid, max_step)
            size = self._size_at_discovery.get(nid, self.total_nodes)

            feats[i, 0] = rank / max_rank              # discovery_order
            feats[i, 1] = step / max_step               # time_step
            feats[i, 2] = size / self.total_nodes        # growth_stage

        return feats

    def augment_features(self, base_features: np.ndarray,
                         node_ids: List[int]) -> np.ndarray:
        """
        Concatenate temporal features to the base feature matrix.

        Parameters
        ----------
        base_features : (N, D) array of existing node features
        node_ids      : list of node IDs matching rows of base_features

        Returns
        -------
        (N, D+3) array
        """
        temporal = self.get_temporal_features(node_ids)
        return np.concatenate([base_features, temporal], axis=1)

    @property
    def num_temporal_features(self) -> int:
        return 3

    def reset(self) -> None:
        """Clear all state for a new tracing run."""
        self._discovery.clear()
        self._step_discovered.clear()
        self._size_at_discovery.clear()
        self._current_step = 0


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Embedding Memory (Exponential Smoothing)
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingMemory:
    """
    Stores per-node embeddings across tracing steps and provides
    exponential moving average (EMA) smoothing.

        smoothed_h(t) = α · h(t)_current + (1 − α) · smoothed_h(t−1)

    This dampens prediction oscillations without adding any learnable
    parameters.
    """

    def __init__(self, embed_dim: int, alpha: float = 0.3):
        """
        Parameters
        ----------
        embed_dim : dimensionality of node embeddings
        alpha     : smoothing factor ∈ (0, 1].
                    Higher → more weight on current embedding.
                    Lower  → more weight on history (more stable).
        """
        self.embed_dim = embed_dim
        self.alpha = alpha
        self._memory: Dict[int, np.ndarray] = {}   # node_id → smoothed embedding
        self._step_count: Dict[int, int] = {}       # how many times each node updated

    def store(self, node_id: int, embedding: np.ndarray) -> None:
        """Store raw embedding (no smoothing). Used for initial insertion."""
        self._memory[node_id] = embedding.copy()
        self._step_count[node_id] = 1

    def recall(self, node_id: int) -> Optional[np.ndarray]:
        """Retrieve the last smoothed embedding, or None if unseen."""
        return self._memory.get(node_id)

    def smooth(self, node_id: int, current_embedding: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing and return the smoothed embedding.
        Also updates the internal memory.
        """
        prev = self._memory.get(node_id)
        if prev is None:
            # First time seeing this node
            self._memory[node_id] = current_embedding.copy()
            self._step_count[node_id] = 1
            return current_embedding.copy()

        smoothed = self.alpha * current_embedding + (1.0 - self.alpha) * prev
        self._memory[node_id] = smoothed
        self._step_count[node_id] = self._step_count.get(node_id, 0) + 1
        return smoothed

    def smooth_batch(self, node_ids: List[int],
                     embeddings: np.ndarray) -> np.ndarray:
        """
        Smooth a batch of embeddings. Returns the smoothed array (N, D).
        """
        result = np.zeros_like(embeddings)
        for i, nid in enumerate(node_ids):
            result[i] = self.smooth(nid, embeddings[i])
        return result

    def compute_drift(self, node_id: int,
                      current_embedding: np.ndarray) -> float:
        """
        L2-norm of the change from the stored embedding to the current one.
        Useful for tracking convergence / stability.
        """
        prev = self._memory.get(node_id)
        if prev is None:
            return float('inf')
        return float(np.linalg.norm(current_embedding - prev))

    def compute_batch_drift(self, node_ids: List[int],
                            embeddings: np.ndarray) -> float:
        """Mean L2 drift across a batch of nodes."""
        drifts = []
        for i, nid in enumerate(node_ids):
            d = self.compute_drift(nid, embeddings[i])
            if d != float('inf'):
                drifts.append(d)
        return float(np.mean(drifts)) if drifts else float('inf')

    def reset(self) -> None:
        """Clear all stored embeddings."""
        self._memory.clear()
        self._step_count.clear()


# ═══════════════════════════════════════════════════════════════════════════
# 3.  GRU Temporal Update (Learnable)
# ═══════════════════════════════════════════════════════════════════════════

class GRUTemporalUpdate(nn.Module):
    """
    A single GRU cell that merges the current GNN embedding with the
    previous time-step embedding:

        h_v^(t) = GRU(h_current, h_previous)

    Applied *after* the SAGE forward pass but *before* the final
    scoring/output layer.

    Parameters
    ----------
    hidden_dim : dimensionality of the GNN hidden embeddings
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(input_size=hidden_dim,
                                   hidden_size=hidden_dim)

    def forward(self, current: torch.Tensor,
                previous: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        current  : (N, hidden_dim) — embeddings from this step's GNN pass
        previous : (N, hidden_dim) — embeddings from the last step
                   (zero-initialised for nodes seen for the first time)

        Returns
        -------
        (N, hidden_dim) — temporally updated embeddings
        """
        return self.gru_cell(current, previous)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Temporal Tracing State (Orchestrator)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TemporalTracingState:
    """
    Orchestrator that ties together temporal features, embedding memory,
    and optionally a GRU update into a single coherent interface for
    use inside a tracing loop.

    Usage inside a tracing step:

        # 1. Register newly discovered nodes
        state.register_new_nodes(new_node_ids, step, len(observed))

        # 2. Get augmented features (base + 3 temporal)
        augmented = state.augment(base_feats, node_ids)

        # 3. After GNN forward pass, smooth embeddings
        smoothed = state.smooth_embeddings(node_ids, raw_embeddings)
    """

    total_nodes: int
    embed_dim:   int       = 50     # must match model hid_feats
    alpha:       float     = 0.3    # EMA smoothing factor
    use_gru:     bool      = False

    # ── Internal state (created in __post_init__) ─────────────────────
    _augmentor:  Optional[TemporalFeatureAugmentor] = field(
        default=None, init=False, repr=False)
    _memory:     Optional[EmbeddingMemory] = field(
        default=None, init=False, repr=False)
    _gru:        Optional[GRUTemporalUpdate] = field(
        default=None, init=False, repr=False)

    # Per-step logging
    prediction_changes: List[bool]  = field(default_factory=list, repr=False)
    embedding_drifts:   List[float] = field(default_factory=list, repr=False)
    _last_prediction:   Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._augmentor = TemporalFeatureAugmentor(self.total_nodes)
        self._memory    = EmbeddingMemory(self.embed_dim, self.alpha)
        if self.use_gru:
            self._gru = GRUTemporalUpdate(self.embed_dim)

    # ── Public interface ──────────────────────────────────────────────

    def register_seed(self, seed_node: int) -> None:
        """Register the seed node at step 0."""
        self._augmentor.register_seed(seed_node)

    def register_new_nodes(self, new_nodes: List[int], step: int,
                           current_graph_size: int) -> None:
        """Register newly discovered nodes at this step."""
        self._augmentor.register_nodes(new_nodes, step, current_graph_size)

    def augment(self, base_features: np.ndarray,
                node_ids: List[int]) -> np.ndarray:
        """
        Append 3 temporal features to the base feature matrix.

        Parameters
        ----------
        base_features : (N, 8) or (N, 10) array
        node_ids      : list of N node IDs

        Returns
        -------
        (N, 11) or (N, 13) array
        """
        return self._augmentor.augment_features(base_features, node_ids)

    def smooth_embeddings(self, node_ids: List[int],
                          embeddings: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing to embeddings. Also computes and logs
        embedding drift.
        """
        drift = self._memory.compute_batch_drift(node_ids, embeddings)
        self.embedding_drifts.append(drift)
        return self._memory.smooth_batch(node_ids, embeddings)

    def gru_update(self, node_ids: List[int],
                   current_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply GRU temporal update if enabled. Otherwise returns
        current_embeddings unchanged.
        """
        if self._gru is None:
            return current_embeddings

        # Build previous embeddings tensor
        N = current_embeddings.shape[0]
        prev = torch.zeros_like(current_embeddings)
        for i, nid in enumerate(node_ids):
            recalled = self._memory.recall(nid)
            if recalled is not None:
                prev[i] = torch.tensor(recalled, dtype=current_embeddings.dtype)

        updated = self._gru(current_embeddings, prev)

        # Store the updated embeddings back
        updated_np = updated.detach().cpu().numpy()
        for i, nid in enumerate(node_ids):
            self._memory.store(nid, updated_np[i])

        return updated

    def log_prediction(self, predicted_source: int) -> None:
        """Track whether the prediction changed from the last step."""
        changed = (self._last_prediction is not None and
                   predicted_source != self._last_prediction)
        self.prediction_changes.append(changed)
        self._last_prediction = predicted_source

    @property
    def prediction_stability(self) -> float:
        """
        Fraction of steps where the prediction did NOT change.
        Higher is more stable. Returns 1.0 if no history.
        """
        if not self.prediction_changes:
            return 1.0
        return 1.0 - (sum(self.prediction_changes) /
                       len(self.prediction_changes))

    @property
    def mean_embedding_drift(self) -> float:
        """Average embedding drift across all logged steps."""
        finite = [d for d in self.embedding_drifts if d != float('inf')]
        return float(np.mean(finite)) if finite else float('inf')

    def summary(self) -> Dict:
        """Return aggregated temporal metrics."""
        return {
            "prediction_stability": round(self.prediction_stability, 4),
            "mean_embedding_drift": round(self.mean_embedding_drift, 6)
                                    if self.mean_embedding_drift != float('inf')
                                    else None,
            "total_prediction_changes": sum(self.prediction_changes),
            "steps_logged":            len(self.prediction_changes),
        }

    def reset(self) -> None:
        """Clear all state for a new tracing run."""
        self._augmentor.reset()
        self._memory.reset()
        self.prediction_changes.clear()
        self.embedding_drifts.clear()
        self._last_prediction = None


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  temporal.py — self-test")
    print("=" * 60)

    # ── 1. TemporalFeatureAugmentor ───────────────────────────────────
    print("\n1. TemporalFeatureAugmentor")
    aug = TemporalFeatureAugmentor(total_nodes=100)
    aug.register_seed(seed_node=42)
    aug.register_nodes([10, 20, 30], step=1, current_graph_size=4)
    aug.register_nodes([50, 60], step=2, current_graph_size=6)

    feats = aug.get_temporal_features([42, 10, 20, 30, 50, 60])
    print(f"   Temporal features shape: {feats.shape}")
    print(f"   Seed node (42) discovery_order: {feats[0, 0]:.3f} "
          f"(expect 0.000)")
    print(f"   Last node (60) discovery_order: {feats[5, 0]:.3f} "
          f"(expect 1.000)")
    assert feats.shape == (6, 3), f"Expected (6, 3), got {feats.shape}"
    assert feats[0, 0] == 0.0, "Seed should have discovery_order=0"
    print("   ✓ TemporalFeatureAugmentor OK")

    # Test augmentation
    base = np.random.randn(6, 8).astype(np.float32)
    augmented = aug.augment_features(base, [42, 10, 20, 30, 50, 60])
    assert augmented.shape == (6, 11), f"Expected (6, 11), got {augmented.shape}"
    print(f"   Augmented shape: {augmented.shape} ✓")

    # ── 2. EmbeddingMemory ────────────────────────────────────────────
    print("\n2. EmbeddingMemory")
    mem = EmbeddingMemory(embed_dim=4, alpha=0.5)

    e1 = np.array([1.0, 0.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0, 0.0])

    s1 = mem.smooth(node_id=0, current_embedding=e1)
    assert np.allclose(s1, e1), "First smooth should return current"
    print(f"   First smooth:  {s1} (expect {e1}) ✓")

    s2 = mem.smooth(node_id=0, current_embedding=e2)
    expected = 0.5 * e2 + 0.5 * e1
    assert np.allclose(s2, expected), f"Expected {expected}, got {s2}"
    print(f"   Second smooth: {s2} (expect {expected}) ✓")

    drift = mem.compute_drift(0, e2)
    print(f"   Drift from stored to e2: {drift:.4f}")
    assert drift > 0, "Drift should be positive"
    print("   ✓ EmbeddingMemory OK")

    # ── 3. GRUTemporalUpdate ──────────────────────────────────────────
    print("\n3. GRUTemporalUpdate")
    gru = GRUTemporalUpdate(hidden_dim=16)
    cur  = torch.randn(5, 16)
    prev = torch.randn(5, 16)
    out  = gru(cur, prev)
    assert out.shape == (5, 16), f"Expected (5, 16), got {out.shape}"
    print(f"   GRU output shape: {out.shape} ✓")

    # Verify gradients flow
    loss = out.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in gru.parameters())
    assert has_grad, "GRU should have gradients"
    print("   Gradients flow ✓")
    print("   ✓ GRUTemporalUpdate OK")

    # ── 4. TemporalTracingState ───────────────────────────────────────
    print("\n4. TemporalTracingState")
    state = TemporalTracingState(total_nodes=50, embed_dim=4, alpha=0.4)
    state.register_seed(seed_node=0)
    state.register_new_nodes([1, 2, 3], step=1, current_graph_size=4)
    state.register_new_nodes([4, 5], step=2, current_graph_size=6)

    # Augment
    base_feats = np.random.randn(6, 8).astype(np.float32)
    node_ids = [0, 1, 2, 3, 4, 5]
    aug_feats = state.augment(base_feats, node_ids)
    assert aug_feats.shape == (6, 11)
    print(f"   Augmented features: {aug_feats.shape} ✓")

    # Smooth
    raw_emb = np.random.randn(6, 4).astype(np.float32)
    smoothed = state.smooth_embeddings(node_ids, raw_emb)
    assert smoothed.shape == raw_emb.shape
    print(f"   Smoothed embeddings: {smoothed.shape} ✓")

    # Second smoothing → drift should be finite
    raw_emb2 = np.random.randn(6, 4).astype(np.float32)
    smoothed2 = state.smooth_embeddings(node_ids, raw_emb2)
    assert state.embedding_drifts[-1] != float('inf')
    print(f"   Embedding drift (step 2): {state.embedding_drifts[-1]:.4f} ✓")

    # Prediction tracking
    state.log_prediction(0)
    state.log_prediction(0)
    state.log_prediction(1)
    state.log_prediction(1)
    # log sequence: None→0 (no change), 0→0 (no change), 0→1 (change), 1→1 (no change)
    # changes = [False, False, True, False] → stability = 1 - 1/4 = 0.75
    print(f"   Prediction stability: {state.prediction_stability:.2f} "
          f"(expect 0.75)")
    assert abs(state.prediction_stability - 0.75) < 1e-6, \
        f"Expected 0.75, got {state.prediction_stability}"

    # Summary
    summary = state.summary()
    print(f"   Summary: {summary}")
    print("   ✓ TemporalTracingState OK")

    print("\n" + "=" * 60)
    print("  ✓ All temporal module self-tests passed.")
    print("=" * 60)
