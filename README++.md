# DeepTrace++ — Enhanced Epidemic Source Tracing with GNN

> **DeepTrace++** is an enhanced version of the original **DeepTrace** pipeline. It adds three new capabilities on top of the original GNN-based superspreader identification framework: enriched node features, confidence-guided adaptive expansion, and temporal (stage-wise) awareness across tracing steps.

---

## Table of Contents
1. [How DeepTrace++ Differs from DeepTrace](#1-how-deeptrace-differs-from-deeptrace)
2. [Repository Structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Running the Pipeline](#4-running-the-pipeline)
5. [Output Files](#5-output-files)
6. [Module Reference](#6-module-reference)

---

## 1. How DeepTrace++ Differs from DeepTrace

### Original DeepTrace (baseline)

The original pipeline works in two stages:
1. **Forward tracing** — expands the observed contact graph G_n using **BFS or DFS** (fixed, blind traversal order)
2. **Backward inference** — runs a 4-layer GraphSAGE GNN on G_n to score each node as a potential superspreader

These two stages operate **independently**. The graph expansion does not use the model's output to decide where to expand next.

```
Observed graph G_n  →  BFS / DFS expansion  →  GNN inference  →  Source prediction
        (fixed order, no feedback)                (memoryless)
```

---

### DeepTrace++ additions

DeepTrace++ introduces **three layered improvements**, each building on the last:

#### Layer 1 — Enriched Node Features (in `graph_data_process.py`)
Two new structural features are added to the original 8:
- `closeness_centrality` — how quickly a node can reach all others
- `norm_dist_to_index` — normalised distance from the highest-degree node (index case proxy)

| Feature set | Dimensions | File |
|---|---|---|
| Original | 8 | `graph_data_process.py` (`ORIGINAL_FEAT_DIM`) |
| Enriched | 10 | `graph_data_process.py` (`ENRICHED_FEAT_DIM`) |
| Temporal | 11 | `graph_data_process.py` (`TEMPORAL_FEAT_DIM`) |

#### Layer 2 — Confidence-Guided Adaptive Expansion (in `confidence.py` + `topk_tracing.py`)

After each GNN inference step, a **confidence score** is computed from the model's output distribution. This score dynamically controls how many frontier nodes are expanded at the next step:

```
High confidence  →  expand fewer nodes  (exploit: the model knows where to look)
Low  confidence  →  expand more  nodes  (explore: the model is uncertain)
```

Three **confidence measures** are available:
- `margin`    — gap between top-1 and top-2 predicted scores
- `entropy`   — how "peaked" the softmax distribution is
- `gap_ratio` — how far the top prediction stands above the mean

Three **adaptive-k strategies** control how confidence maps to expansion size:
- `linear`      — proportional decrease
- `exponential` — steep drop-off at high confidence
- `step`        — three discrete tiers (high / medium / low)

When confidence is very low (< 0.1), the system falls back to BFS/DFS order to avoid random exploration.

#### Layer 3 — Temporal (Stage-Wise) Awareness (in `temporal.py` + `topk_tracing.py`)

The original GNN treats every graph snapshot as independent. DeepTrace++ adds **memory across tracing steps**:

1. **Temporal features** appended to each node's feature vector:
   - `discovery_order` — when was this node discovered (normalised rank)
   - `time_step` — at which tracing step did it enter G_n
   - `growth_stage` — how "full" was the observed graph when this node was added

2. **Embedding memory** — after each GNN pass, node embeddings are smoothed using an exponential moving average (EMA) with factor α:
   ```
   h_v(t) = α · h_current + (1 - α) · h_previous
   ```
   This prevents predictions from oscillating wildly between steps.

3. **GRU recurrent update** (optional) — a learnable GRU cell that merges the current embedding with the previous one, enabling the model to learn *how* to integrate history.

---

### Side-by-side comparison

| Capability | DeepTrace | DeepTrace++ |
|---|---|---|
| Graph expansion | BFS / DFS only | BFS / DFS / Top-K / **Adaptive-k** |
| Expansion guidance | None (fixed order) | **GNN confidence scores** |
| Exploration strategy | Static | **Dynamic (exploit when confident, explore when not)** |
| Node features | 8 | **8 → 10 (enriched) → 11 (temporal)** |
| Model memory across steps | None | **EMA embedding smoothing** |
| Learnable recurrence | None | **Optional GRU cell (TemporalSAGE)** |
| Prediction stability | Can oscillate | **Tracked and stabilised** |
| Convergence detection | None | **Logged — step where prediction stabilises** |
| Output logs | None | **CSV traces for confidence, temporal, comparison** |

---

## 2. Repository Structure

```
deeptrace/
│
├── README++.md                  ← This file
├── README.md                    ← Original README
│
│── Core pipeline (original DeepTrace)
├── model.py                     ← GraphSAGE model, training, evaluation
├── graph_data_process.py        ← Node feature computation
├── label_list_process.py        ← Label generation (DS probabilities)
├── bfs_tracing.py               ← Original BFS tracing
├── dfs_tracing.py               ← Original DFS tracing
├── cal_max_min_ds.py            ← MLE geometric probability heuristic
├── average_ds.py                ← DS probability simulations
│
│── New DeepTrace++ modules
├── confidence.py                ← [NEW] Confidence measures + adaptive-k + ConfidenceTracker
├── temporal.py                  ← [NEW] Temporal features + EmbeddingMemory + GRUTemporalUpdate
├── topk_tracing.py              ← [MODIFIED] Adaptive + temporal tracing loops
│                                          compare_strategies() unified runner
│
│── Data
├── data/                        ← Hong Kong and Taiwan COVID-19 contact data
│
│── Evaluation utilities
├── hop_error.py                 ← Hop distance error metrics
├── average_error.py             ← Averaged error reporting
├── contact_tracing_involve.py   ← Tracing involvement analysis
└── rumor_centrality.py          ← Rumor centrality baseline
```

---

## 3. Installation

### Requirements
- Python **3.11 or 3.12** recommended (DGL has no PyPI wheel for Python 3.13)
- PyTorch ≥ 2.0
- Deep Graph Library (DGL)
- NetworkX, NumPy, Pandas, Matplotlib, Seaborn

### Install steps

```bash
# 1. Install Python 3.11 or 3.12 from https://python.org

# 2. Install standard dependencies
pip install torch networkx pandas numpy matplotlib seaborn

# 3. Install DGL (must match your Python version)
#    For CPU-only:
pip install dgl -f https://data.dgl.ai/wheels/repo.html

#    If pip fails, use conda:
conda install -c dglteam dgl

# 4. (Optional) install from requirements if provided
pip install -r requirements.txt
```

> **Note:** The confidence and temporal modules (`confidence.py`, `temporal.py`) do **not** require DGL.  
> You can run their self-tests on any Python version with just `numpy`, `pandas`, and `torch`.

---

## 4. Running the Pipeline

### Step 1 — Verify standalone modules (no DGL needed)

```bash
cd deeptrace/

# Test confidence measures + adaptive-k
python confidence.py

# Test temporal features + embedding memory + GRU
python temporal.py
```

Both should print `✓ All ... self-tests passed.`

---

### Step 2 — Train the GNN model (original DeepTrace)

```bash
python model.py
```

This trains a 4-layer GraphSAGE on synthetic epidemic trees and evaluates the R² score. Outputs predictions to `eval_position.csv`.

---

### Step 3 — Run the full DeepTrace++ tracing comparison

```bash
python topk_tracing.py
```

This runs **all strategies** on a test graph and saves results:

| Strategy | Description |
|---|---|
| `bfs` | Original BFS expansion (baseline) |
| `dfs` | Original DFS expansion (baseline) |
| `top_k (k=1,3,5)` | GNN-guided top-k expansion |
| `adaptive` | Confidence-guided adaptive-k expansion |
| `temporal_adaptive` | Temporal + EMA smoothing + adaptive-k |
| `temporal_adaptive_gru` | Temporal + GRU recurrence + adaptive-k |

---

### Step 4 — Run on real COVID-19 data

```bash
python average_ds.py       # Run DS probability simulations
python hop_error.py        # Compute hop-error metrics across the dataset
```

---

### Custom usage — use adaptive tracing in your own code

```python
import networkx as nx
from topk_tracing import adaptive_tracing, temporal_adaptive_tracing
from model import SAGE

# Load your trained model
model = SAGE(in_feats=8, hid_feats=50, out_feats=1)
# model.load_state_dict(torch.load("your_model.pt"))

G = nx.read_edgelist("your_graph.edgelist", nodetype=int)
true_source = 0  # ground truth (for evaluation)

# — Confidence-guided adaptive expansion —
metrics, tracker = adaptive_tracing(
    G, true_source, model,
    k_min=1, k_max=8,
    confidence_method="margin",   # or "entropy" / "gap_ratio"
    k_strategy="linear",          # or "exponential" / "step"
)
print(tracker.summary())

# — Temporal + adaptive expansion —
from model import TemporalSAGE
t_model = TemporalSAGE(in_feats=11, hid_feats=50, out_feats=1, use_gru=True)

metrics, tracker, t_state = temporal_adaptive_tracing(
    G, true_source, t_model,
    k_min=1, k_max=8,
    embed_dim=50,
    alpha=0.3,        # EMA smoothing factor
    use_gru=True,
)
print(t_state.summary())
```

---

## 5. Output Files

All logs are written to `adaptive_tracing_logs/`:

| File | Contents |
|---|---|
| `comparison_results.csv` | One row per strategy: steps, nodes explored, hop error, correctness, wall time |
| `confidence_trace_<name>.csv` | Per-step: confidence score, k used, nodes explored, hop error, predicted source |
| `temporal_trace_<name>.csv` | Per-step: confidence, k, nodes explored, hop error, embedding drift |
| `temporal_comparison.csv` | Per-run: prediction stability, mean embedding drift, convergence step |

`topk_vs_bfs_dfs.csv` is also written at the root (backward-compatible).

---

## 6. Module Reference

### `confidence.py` — Confidence measures and adaptive-k

| Function / Class | Description |
|---|---|
| `compute_confidence(scores, method)` | Scalar ∈ [0,1] from GNN prediction scores |
| `adaptive_k(confidence, k_min, k_max, strategy)` | Maps confidence → integer expansion budget |
| `ConfidenceTracker` | Logs per-step: confidence, k, nodes, hop error. Detects convergence step |

### `temporal.py` — Temporal state management

| Function / Class | Description |
|---|---|
| `TemporalFeatureAugmentor` | Adds `discovery_order`, `time_step`, `growth_stage` features (8→11 dims) |
| `EmbeddingMemory` | EMA smoothing of node embeddings across steps; tracks drift |
| `GRUTemporalUpdate` | Single GRU cell merging current + previous embedding |
| `TemporalTracingState` | Orchestrates the above + logs `prediction_stability`, `embedding_drift` |

### `topk_tracing.py` — Tracing strategies

| Function | Description |
|---|---|
| `top_k_tracing()` | GNN-guided top-k expansion with fixed k |
| `adaptive_tracing()` | Confidence-guided adaptive k (Layer 2) |
| `temporal_adaptive_tracing()` | Full temporal + confidence pipeline (Layer 3) |
| `bfs_tracing_v2()` | BFS baseline with unified metric logging |
| `dfs_tracing_v2()` | DFS baseline with unified metric logging |
| `compare_strategies()` | Runs all strategies on one graph, returns comparison DataFrame |

### `model.py` — GNN models

| Class | Description |
|---|---|
| `SAGE` | Original 4-layer GraphSAGE (in_feats=8) |
| `WeightedSAGE` | DeepTrace++ edge-weighted variant |
| `TemporalSAGE` | Temporal-aware SAGE (in_feats=11) with optional GRU recurrence and `get_embeddings()` |

---

## Credits

- **Original DeepTrace**: GNN-based superspreader identification via maximum-likelihood estimation on contact networks
- **DeepTrace++**: Adaptive exploration and temporal awareness extensions implemented on top of the original pipeline
- COVID-19 data: Hong Kong (Nature Medicine, 2020) and Taiwan CDC datasets
