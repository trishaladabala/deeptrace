"""
confidence.py — Confidence-Guided Adaptive Expansion for DeepTrace++
=====================================================================
Provides confidence measures computed from GNN inference scores and
adaptive expansion-budget functions that map confidence → k.

Key API
-------
* compute_confidence(scores, method)  →  float ∈ [0, 1]
* adaptive_k(confidence, k_min, k_max, strategy) → int
* ConfidenceTracker  — per-step metric accumulator with convergence detection

The module is intentionally decoupled from the GNN model and graph
expansion logic so that it can be used with any scoring backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1.  Confidence measures
# ---------------------------------------------------------------------------

def compute_confidence(
    scores: Dict[int, float],
    method: str = "margin",
) -> float:
    """
    Compute a scalar confidence ∈ [0, 1] from per-node GNN scores.

    Parameters
    ----------
    scores : {node_id: score}  — raw output of _gnn_scores().
    method : one of
        'margin'    — normalised gap between top-1 and top-2 scores
        'entropy'   — 1 − H(softmax(scores)) / log(N)
        'gap_ratio' — (top1 − mean) / (max − min + ε)

    Returns
    -------
    float in [0, 1].  Higher → more confident.
    """
    if not scores or len(scores) < 2:
        return 0.0  # too few nodes to judge

    vals = np.array(list(scores.values()), dtype=np.float64)

    if method == "margin":
        return _confidence_margin(vals)
    elif method == "entropy":
        return _confidence_entropy(vals)
    elif method == "gap_ratio":
        return _confidence_gap_ratio(vals)
    else:
        raise ValueError(f"Unknown confidence method: {method!r}")


def _confidence_margin(vals: np.ndarray) -> float:
    """
    Margin confidence:  (top1 − top2) / (|top1| + ε)

    When the top prediction dominates, the margin is large → high confidence.
    When scores are bunched together, margin ≈ 0 → low confidence.
    """
    sorted_vals = np.sort(vals)[::-1]            # descending
    top1, top2 = sorted_vals[0], sorted_vals[1]
    eps = 1e-9
    raw = (top1 - top2) / (abs(top1) + eps)
    return float(np.clip(raw, 0.0, 1.0))


def _confidence_entropy(vals: np.ndarray) -> float:
    """
    Entropy-based confidence:  1 − H(softmax(scores)) / log(N)

    A peaked distribution (one clear winner) → low entropy → high confidence.
    A uniform distribution (all equally likely) → max entropy → low confidence.
    """
    n = len(vals)
    if n <= 1:
        return 0.0

    # Numerically stable softmax
    shifted = vals - vals.max()
    exp_vals = np.exp(shifted)
    probs = exp_vals / (exp_vals.sum() + 1e-12)

    # Shannon entropy (using natural log)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = math.log(n)

    if max_entropy < 1e-12:
        return 0.0

    normalised = entropy / max_entropy          # ∈ [0, 1]
    return float(np.clip(1.0 - normalised, 0.0, 1.0))


def _confidence_gap_ratio(vals: np.ndarray) -> float:
    """
    Gap-ratio confidence:  (top1 − mean) / (max − min + ε)

    Measures how far the highest score stands out from the average
    relative to the overall score range.
    """
    top1 = vals.max()
    mean_val = vals.mean()
    val_range = vals.max() - vals.min()
    eps = 1e-9
    raw = (top1 - mean_val) / (val_range + eps)
    return float(np.clip(raw, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 2.  Adaptive k functions
# ---------------------------------------------------------------------------

def adaptive_k(
    confidence: float,
    k_min: int = 1,
    k_max: int = 10,
    strategy: str = "linear",
    alpha: float = 3.0,
) -> int:
    """
    Map a confidence score ∈ [0, 1] to an integer expansion budget k.

    High confidence → small k  (exploit: expand conservatively)
    Low  confidence → large k  (explore: expand aggressively)

    Parameters
    ----------
    confidence : float ∈ [0, 1]
    k_min      : smallest expansion budget (high-confidence regime)
    k_max      : largest  expansion budget (low-confidence regime)
    strategy   : 'linear' | 'exponential' | 'step'
    alpha      : steepness for the exponential strategy (ignored otherwise)

    Returns
    -------
    int in [k_min, k_max]
    """
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if strategy == "linear":
        k = k_max - confidence * (k_max - k_min)

    elif strategy == "exponential":
        k = k_min + (k_max - k_min) * math.exp(-alpha * confidence)

    elif strategy == "step":
        if confidence > 0.7:
            k = k_min
        elif confidence > 0.3:
            k = (k_min + k_max) / 2.0
        else:
            k = k_max

    else:
        raise ValueError(f"Unknown k-strategy: {strategy!r}")

    return int(np.clip(round(k), k_min, k_max))


# ---------------------------------------------------------------------------
# 3.  ConfidenceTracker — per-step metric accumulator
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceTracker:
    """
    Accumulates per-step metrics during confidence-guided tracing.

    Attributes
    ----------
    steps           : list of step indices (1-based)
    confidence      : confidence score at each step
    k_values        : expansion budget used at each step
    nodes_explored  : cumulative nodes in G_n at each step
    hop_errors      : hop distance between predicted and true source
    predicted_source: the node predicted as source at each step
    """

    steps:             List[int]   = field(default_factory=list)
    confidence:        List[float] = field(default_factory=list)
    k_values:          List[int]   = field(default_factory=list)
    nodes_explored:    List[int]   = field(default_factory=list)
    hop_errors:        List[int]   = field(default_factory=list)
    predicted_source:  List[int]   = field(default_factory=list)

    # ── Recording ─────────────────────────────────────────────────────────

    def record(
        self,
        step:       int,
        conf:       float,
        k:          int,
        n_explored: int,
        hop_err:    int,
        pred_src:   int,
    ) -> None:
        """Append one step of metrics."""
        self.steps.append(step)
        self.confidence.append(conf)
        self.k_values.append(k)
        self.nodes_explored.append(n_explored)
        self.hop_errors.append(hop_err)
        self.predicted_source.append(pred_src)

    # ── Convergence detection ─────────────────────────────────────────────

    @property
    def convergence_step(self) -> Optional[int]:
        """
        First step at which the predicted source stabilised for at least
        3 consecutive steps.  Returns None if convergence was never reached.
        """
        window = 3
        if len(self.predicted_source) < window:
            return None
        for i in range(len(self.predicted_source) - window + 1):
            if len(set(self.predicted_source[i : i + window])) == 1:
                return self.steps[i]
        return None

    # ── Export ─────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame of per-step metrics."""
        return pd.DataFrame({
            "step":             self.steps,
            "confidence":       self.confidence,
            "k":                self.k_values,
            "nodes_explored":   self.nodes_explored,
            "hop_error":        self.hop_errors,
            "predicted_source": self.predicted_source,
        })

    def summary(self) -> Dict:
        """Return aggregated metrics as a dict."""
        return {
            "total_steps":        len(self.steps),
            "mean_confidence":    float(np.mean(self.confidence)) if self.confidence else 0.0,
            "std_confidence":     float(np.std(self.confidence))  if self.confidence else 0.0,
            "mean_k":             float(np.mean(self.k_values))   if self.k_values   else 0.0,
            "final_hop_error":    self.hop_errors[-1]             if self.hop_errors  else None,
            "convergence_step":   self.convergence_step,
            "total_nodes_explored": self.nodes_explored[-1]       if self.nodes_explored else 0,
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Test confidence measures ──────────────────────────────────────────
    uniform_scores = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    peaked_scores  = {0: 10.0, 1: 1.0, 2: 0.5, 3: 0.1, 4: 0.2}

    for method in ("margin", "entropy", "gap_ratio"):
        c_uniform = compute_confidence(uniform_scores, method)
        c_peaked  = compute_confidence(peaked_scores,  method)
        print(f"  {method:12s}  uniform={c_uniform:.4f}  peaked={c_peaked:.4f}")
        assert c_peaked > c_uniform, f"{method}: peaked should be more confident"

    # ── Test adaptive_k ───────────────────────────────────────────────────
    for strategy in ("linear", "exponential", "step"):
        k_low  = adaptive_k(0.0,  k_min=1, k_max=10, strategy=strategy)
        k_high = adaptive_k(1.0,  k_min=1, k_max=10, strategy=strategy)
        print(f"  {strategy:12s}  k(conf=0)={k_low}  k(conf=1)={k_high}")
        assert k_low >= k_high, f"{strategy}: low confidence should give higher k"

    # ── Test ConfidenceTracker ────────────────────────────────────────────
    tracker = ConfidenceTracker()
    for i in range(1, 8):
        tracker.record(step=i, conf=0.1*i, k=10-i, n_explored=i*5,
                       hop_err=max(0, 5-i), pred_src=42 if i >= 3 else i)
    print(f"\n  Tracker summary: {tracker.summary()}")
    print(f"  Convergence step: {tracker.convergence_step}")
    print(f"\n  DataFrame:\n{tracker.to_dataframe().to_string(index=False)}")

    print("\n✓ All confidence module self-tests passed.")
