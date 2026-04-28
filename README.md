# DeepTrace / DeepTrace++

A Graph Neural Network framework for epidemic source tracing and superspreader identification in contact networks.

DeepTrace estimates the probability of each node being a superspreader by computing the maximum-likelihood estimator and leveraging the likelihood structure to configure the training set with topological features of smaller epidemic network datasets.

DeepTrace++ extends the original pipeline with enriched node features, edge-weighted message passing, confidence-guided adaptive tracing, and temporal awareness across tracing steps.

---

## Software Dependencies

- Python 3.11 or 3.12 (DGL does not have a PyPI wheel for Python 3.13)
- PyTorch >= 2.0
- Deep Graph Library (DGL)
- NetworkX
- NumPy, Pandas, Matplotlib, Seaborn

```bash
pip install torch networkx pandas numpy matplotlib seaborn

# DGL (CPU-only):
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# If pip fails, use conda:
conda install -c dglteam dgl
```

---

## Repository Structure

```
deeptrace/
├── model.py                   ← GNN models (SAGE, WeightedSAGE, TemporalSAGE), training, evaluation
├── graph_data_process.py      ← Node feature computation (8, 10, or 11 features)
├── label_list_process.py      ← Label generation (DS probabilities)
├── cal_max_min_ds.py          ← MLE geometric probability heuristic
├── confidence.py              ← Confidence measures + adaptive-k logic
├── temporal.py                ← Temporal features + EmbeddingMemory + GRU update
├── topk_tracing.py            ← Adaptive + temporal tracing loops, strategy comparison
├── evaluation.py              ← Full evaluation across network topologies
├── evaluate_real_world.py     ← Real-world Hong Kong / Taiwan dataset evaluation
├── bfs_tracing.py             ← Original BFS tracing
├── dfs_tracing.py             ← Original DFS tracing
├── hop_error.py               ← Hop distance error metrics
├── average_ds.py              ← DS probability simulations
├── average_error.py           ← Averaged error reporting
├── rumor_centrality.py        ← Rumor centrality baseline
├── contact_tracing_involve.py ← Tracing involvement analysis
├── data/                      ← Hong Kong and Taiwan COVID-19 contact data
└── evaluation_results/        ← Saved evaluation outputs
```

---

## Running the Codebase

### 1. Verify standalone modules (no DGL needed)

```bash
cd deeptrace/

python confidence.py
python temporal.py
```

Both scripts run built-in self-tests and should print `✓ All ... self-tests passed.`

### 2. Train the GNN model and run evaluations

```bash
python model.py
```

Trains a 4-layer GraphSAGE on synthetic epidemic trees. Also runs the ablation study (original vs enriched features) and the edge-weight comparison (unweighted vs weighted variants). Outputs predictions to CSV files including `eval_position_ER_new.csv`, `ablation_study.csv`, and `edge_weight_comparison.csv`.

### 3. Run the full evaluation across network topologies

```bash
python evaluation.py
```

Compares DeepTrace, DT++ enriched, DT++ weighted, and the full DeepTrace++ model across Erdos-Renyi, Barabasi-Albert, Watts-Strogatz, Regular, and Stochastic Block Model networks. Measures top-k accuracy, bias, hop error, first detection time, and wall time. Results are saved to `evaluation_results/`.

### 4. Run on real-world COVID-19 data

```bash
python evaluate_real_world.py
```

Trains models on synthetic data and evaluates them on Hong Kong and Taiwan COVID-19 contact tracing datasets located in `data/`. Results are saved to `evaluation_results/real_world_results.csv`.

### 5. Run the adaptive tracing comparison

```bash
python topk_tracing.py
```

Runs all tracing strategies on a test graph and saves results to `adaptive_tracing_logs/`:

| Strategy | Description |
|---|---|
| `bfs` | Original BFS expansion (baseline) |
| `dfs` | Original DFS expansion (baseline) |
| `top_k (k=1,3,5)` | GNN-guided top-k expansion |
| `adaptive` | Confidence-guided adaptive-k expansion |
| `temporal_adaptive` | Temporal + EMA smoothing + adaptive-k |
| `temporal_adaptive_gru` | Temporal + GRU recurrence + adaptive-k |

### 6. Run hop error and DS probability simulations

```bash
python hop_error.py
python average_ds.py
```

---

## Data Processing

Node features are computed in `graph_data_process.py`. The input is a `networkx.Graph` representing an epidemic tree:

- **Original features (8):** node count, degree ratio, average degree ratio, infected neighbour proportion, infected degree proportion, boundary distance ratio, layer rate, layer number
- **Enriched features (10):** adds closeness centrality and normalised distance to the index node
- **Temporal features (11):** adds discovery order, time step, and growth stage

Labels (superspreader probabilities) are generated in `label_list_process.py` using the geometric mean of the maximum and minimum DS probabilities.

---

## GNN Models

| Class | Features | Description |
|---|---|---|
| `SAGE` | 8 or 10 | Original 4-layer GraphSAGE with LSTM aggregator |
| `WeightedSAGE` | 8 or 10 | Edge-weighted variant using degree-based weights |
| `TemporalSAGE` | 11 | Temporal-aware SAGE with optional GRU recurrence |

---

## Epidemic Data

- **Hong Kong** COVID-19 cluster data (February 2020) from the Nature Medicine paper "Clustering and superspreading potential of SARS-CoV-2 infections in Hong Kong" (Adam, Dillon C, and Wu, Peng, et al., 2020).
- **Hong Kong** raw pandemic data (January 31, 2022 – February 3, 2022) from the Hong Kong government open data portal: https://data.gov.hk/en-data/dataset/hk-dh-chpsebcddr-novel-infectious-agent
- **Taiwan** raw pandemic data (March 19, 2022 – April 1, 2022) from the Taiwan Centers of Disease Control: https://www.cdc.gov.tw/En

Processed data is in the `data/` folder.
