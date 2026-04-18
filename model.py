import dgl
import dgl.nn as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from graph_data_process import TreeDataProcess, ORIGINAL_FEAT_DIM, ENRICHED_FEAT_DIM, TEMPORAL_FEAT_DIM
import networkx as nx
import numpy as np
from dgl.convert import from_networkx
from dgl.convert import to_networkx
from cal_max_min_ds import CalMaxMinDS
from math import sqrt
from math import floor
import matplotlib.pyplot as plt
import random
import pandas as pd
import itertools
import time


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='lstm')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='lstm')
        self.conv3 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='lstm')
        self.conv4 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='lstm')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        h = self.conv3(graph, h)
        h = F.relu(h)
        h = self.conv4(graph, h)
        return h


# ---------------------------------------------------------------------------
# Edge weight utilities  —  DeepTrace++
# ---------------------------------------------------------------------------

def assign_edge_weights(dgl_graph, nx_graph: nx.Graph = None,
                        mode: str = 'degree') -> None:
    """
    Compute and store a scalar weight w_uv on every edge of *dgl_graph*.
    Weights are normalised per destination node so that Σ_u w_uv = 1
    (row-softmax), making them valid probability-like attention scalars.

    Three weight modes
    ------------------
    'degree' (default)
        w_uv ∝ 1 / degree(u)  — nodes with fewer connections carry
        proportionally stronger signal per edge (mirrors the DeepTrace
        likelihood intuition that low-degree hubs are more informative).

    'uniform'
        w_uv = 1 / |N(v)|  — identical to the original unweighted SAGE
        mean aggregator; useful as a controlled baseline.

    'random'
        w_uv ~ Uniform(0,1), then row-softmax normalised — used to
        sanity-check that *any* weighting outperforms a random one.

    Parameters
    ----------
    dgl_graph : DGLGraph  (modified in-place)
    nx_graph  : optional source NetworkX graph for degree look-up
    mode      : 'degree' | 'uniform' | 'random'
    """
    num_edges = dgl_graph.num_edges()
    src, dst  = dgl_graph.edges()

    if mode == 'uniform':
        raw = th.ones(num_edges, dtype=th.float32)

    elif mode == 'random':
        raw = th.rand(num_edges, dtype=th.float32)

    else:  # 'degree'
        # Use the DGL graph's in-degree as a proxy when nx_graph absent
        if nx_graph is not None:
            degree_map = dict(nx_graph.degree())
            # Map DGL node ids → nx degree (assumes integer labels preserved)
            src_deg = th.tensor(
                [degree_map.get(int(s), 1) for s in src.tolist()],
                dtype=th.float32
            )
        else:
            src_deg = dgl_graph.in_degrees(src).float().clamp(min=1)
        raw = 1.0 / src_deg  # inverse degree

    # ── Row-softmax per destination node so Σ_u w_uv = 1 ────────────────
    # Segment-wise softmax: for each dst node, normalise incoming weights.
    n_nodes   = dgl_graph.num_nodes()
    # Scatter-max for numerical stability
    max_per_dst = th.zeros(n_nodes, dtype=th.float32).scatter_reduce(
        0, dst, raw, reduce='amax', include_self=True
    )
    shifted   = raw - max_per_dst[dst]          # subtract max for stability
    exp_w     = shifted.exp()
    sum_per_dst = th.zeros(n_nodes, dtype=th.float32).scatter_add(
        0, dst, exp_w
    ).clamp(min=1e-9)
    weights   = exp_w / sum_per_dst[dst]         # normalised ∈ (0,1)

    dgl_graph.edata['w'] = weights.unsqueeze(1)  # shape [E, 1]


# ---------------------------------------------------------------------------
# WeightedSAGE  —  DeepTrace++
# ---------------------------------------------------------------------------

class WeightedMessagePassing(nn.Module):
    """
    A single weighted graph-conv layer.

    Aggregation:
        AGG(v) = Σ_{u ∈ N(v)}  w_uv · h_u

    where w_uv is read from *graph.edata['w']* (set by assign_edge_weights).
    The aggregated neighbourhood vector is then concatenated with the self
    embedding and passed through a linear layer — identical structure to
    SAGEConv 'mean' but with learned (or pre-computed) edge weights.

        h_v^{(l+1)} = W · [h_v^{(l)} ‖ AGG(v)] + b
    """
    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.linear = nn.Linear(in_feats * 2, out_feats)

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            # m_{u→v} = w_uv * h_u
            graph.apply_edges(
                lambda edges: {'m': edges.data['w'] * edges.src['h']}
            )
            # AGG(v) = Σ m_{u→v}
            graph.update_all(
                dgl.function.copy_e('m', 'm'),
                dgl.function.sum('m', 'agg')
            )
            agg = graph.ndata['agg']             # [N, in_feats]
            out = self.linear(th.cat([h, agg], dim=1))
            return out


class WeightedSAGE(nn.Module):
    """
    Four-layer GraphSAGE with *weighted* message passing (DeepTrace++).

    Drop-in replacement for SAGE: same constructor signature, same
    forward(graph, inputs) interface. Requires that the graph's edata
    contains 'w' (set via assign_edge_weights before calling forward).

    Parameters
    ----------
    in_feats, hid_feats, out_feats : identical semantics to SAGE
    """
    def __init__(self, in_feats: int, hid_feats: int, out_feats: int):
        super().__init__()
        self.wmp1 = WeightedMessagePassing(in_feats,   hid_feats)
        self.wmp2 = WeightedMessagePassing(hid_feats,  hid_feats)
        self.wmp3 = WeightedMessagePassing(hid_feats,  hid_feats)
        self.wmp4 = WeightedMessagePassing(hid_feats,  out_feats)

    def forward(self, graph, inputs):
        if 'w' not in graph.edata:
            # Safety fallback: assign uniform weights if caller forgot
            assign_edge_weights(graph, mode='uniform')
        h = F.relu(self.wmp1(graph, inputs))
        h = F.relu(self.wmp2(graph, h))
        h = F.relu(self.wmp3(graph, h))
        h = self.wmp4(graph, h)
        return h


# ---------------------------------------------------------------------------
# TemporalSAGE  —  DeepTrace++ temporal awareness
# ---------------------------------------------------------------------------

class TemporalSAGE(nn.Module):
    """
    GraphSAGE variant that accepts temporal-augmented features and
    optionally applies a GRU recurrent update on hidden embeddings.

    Drop-in compatible with SAGE: same forward(graph, inputs) signature.
    The only difference is that `in_feats` should be TEMPORAL_FEAT_DIM (11)
    instead of ORIGINAL_FEAT_DIM (8), and the model exposes a `get_embeddings`
    method that returns the intermediate hidden representations (before the
    output layer) for use with the embedding memory / smoothing system.

    Parameters
    ----------
    in_feats  : input feature dimension (11 for temporal, 8 for original)
    hid_feats : hidden layer dimension
    out_feats : output dimension (1 for superspreader scoring)
    use_gru   : if True, insert a GRU cell between conv3 and conv4
                that merges current embeddings with previous-step embeddings.
                The caller must provide `prev_hidden` to forward().
    """

    def __init__(self, in_feats: int, hid_feats: int, out_feats: int,
                 use_gru: bool = False):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='lstm')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='lstm')
        self.conv3 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='lstm')
        self.conv4 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='lstm')

        self.use_gru = use_gru
        if use_gru:
            self.gru_cell = nn.GRUCell(
                input_size=hid_feats, hidden_size=hid_feats)

    def forward(self, graph, inputs, prev_hidden=None):
        """
        Parameters
        ----------
        graph       : DGL graph
        inputs      : (N, in_feats) node features
        prev_hidden : (N, hid_feats) optional previous hidden state
                      for GRU update.  Ignored if use_gru=False.

        Returns
        -------
        logits : (N, out_feats) — predictions
        """
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        h = self.conv3(graph, h)
        h = F.relu(h)

        # Optional GRU recurrent update before the output layer
        if self.use_gru and prev_hidden is not None:
            h = self.gru_cell(h, prev_hidden)

        h = self.conv4(graph, h)
        return h

    def get_embeddings(self, graph, inputs, prev_hidden=None):
        """
        Run the forward pass but return the *hidden embeddings* (before
        the final output layer) instead of the logits.  Used by the
        embedding memory / smoothing system.

        Returns
        -------
        embeddings : (N, hid_feats)
        logits     : (N, out_feats)
        """
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        h = self.conv3(graph, h)
        h = F.relu(h)

        if self.use_gru and prev_hidden is not None:
            h = self.gru_cell(h, prev_hidden)

        embeddings = h.clone()           # save pre-output embeddings
        logits     = self.conv4(graph, h)
        return embeddings, logits


def evaluate(model, graph, features, labels):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        loss = F.mse_loss(logits.squeeze(-1), labels)
        r2_loss = R2_score(logits.squeeze(-1), labels)
        return r2_loss


def evaluate_position(model, graph, features, labels):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        real_labels_list = labels.numpy().tolist()
        max_real_idx = real_labels_list.index(max(real_labels_list))

        eval_label_list = logits.numpy().tolist()
        pos_eval_val = eval_label_list[max_real_idx]
        eval_label_list_sort = sorted(eval_label_list, reverse=True)
        pos_eval_val_idx = eval_label_list_sort.index(pos_eval_val)

    return pos_eval_val_idx


def evaluate_prob(model, graph, features, labels):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        real_labels_list = labels.numpy().tolist()
        eval_label_list = list(itertools.chain.from_iterable(logits.numpy().tolist()))
        node_index = range(len(real_labels_list))
        real_labels_dict = dict(zip(node_index, real_labels_list))
        eval_labels_dict = dict(zip(node_index, eval_label_list))
    return real_labels_list, eval_label_list


def _sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def data_process_for_single_tree(tree: nx.Graph, enriched: bool = False) -> dgl:
    graph = TreeDataProcess(tree)
    unift_node_list = graph.get_uninfected_node_list()
    graph_nfeature = graph.nfeature_process(enriched=enriched)
    tree.remove_nodes_from(list(unift_node_list))

    geo_permute_prob_list = []
    for node in nx.nodes(tree):
        if node not in set(unift_node_list):
            cal_ds = CalMaxMinDS(tree, unift_node_list, node)
            max_permute_prob = cal_ds.cal_max_ds()
            min_permute_prob = cal_ds.cal_min_ds()
            geo_permute_prob = sqrt(max_permute_prob * min_permute_prob)
            geo_permute_prob_list.append(geo_permute_prob)

    labels = geo_permute_prob_list
    labels = th.tensor(labels)

    graph_nfeature_arr = []
    for k, v in graph_nfeature.items():
        if k not in set(unift_node_list):
            # Base 8 features
            feature_row = [
                v["node_num"], v["degree_per"], v["degree_per_aver"],
                v["inft_ndegree_per"], v["inft_alldegree_per"],
                v["distance_per"], v["layer_rate"], v["layer_num"]
            ]
            # Append enriched features if requested
            if enriched:
                feature_row.append(v["closeness_centrality"])
                feature_row.append(v["norm_dist_to_index"])
            graph_nfeature_arr.append(feature_row)

    g_nfeature = th.tensor(graph_nfeature_arr)
    dgl_tree = from_networkx(tree)
    dgl_tree.ndata["labels"] = labels
    dgl_tree.ndata["feat"] = g_nfeature
    return dgl_tree


def train_data_process(tree_num: int, node_num: int, enriched: bool = False):
    nxtree_list = []
    for i in range(tree_num):
        degree = floor(node_num * 3 / 20)
        ER = nx.random_graphs.watts_strogatz_graph(node_num, degree, 0.3)
        tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
        nxtree_list.append(tree_ka)

    dgl_tree_list = []
    for tree in nxtree_list:
        dgl_tree = data_process_for_single_tree(tree, enriched=enriched)
        dgl_tree_list.append(dgl_tree)
    return dgl.batch(dgl_tree_list)


def test_data_process(tree_num: int, node_num: int, enriched: bool = False):
    nxtree_list = []
    for i in range(tree_num):
        degree = floor(node_num * 3 / 20)
        ER = nx.random_graphs.watts_strogatz_graph(node_num, degree, 0.3)
        tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
        nxtree_list.append(tree_ka)

    dgl_tree_list = []
    for tree in nxtree_list:
        dgl_tree = data_process_for_single_tree(tree, enriched=enriched)
        dgl_tree_list.append(dgl_tree)
    return dgl.batch(dgl_tree_list)


def R2_score(eval_label, real_val):
    len_label = len(eval_label)
    real_val_aver = sum(real_val)/len_label
    sum_diff1 = 0
    sum_diff2 = 0
    for i in range(len_label):
        sum_diff1 = sum_diff1 + (eval_label[i] - real_val[i])**2
        sum_diff2 = sum_diff2 + (real_val[i] - real_val_aver)**2
    res = 1 - (sum_diff1 / sum_diff2)
    return res


def gnn_test_mse(train_patch_tree: dgl, test_patch_tree: dgl):
    train_features_dim = train_patch_tree.ndata["feat"].shape[1]
    train_node_features = train_patch_tree.ndata["feat"]
    train_node_labels = train_patch_tree.ndata["labels"]

    test_node_features = test_patch_tree.ndata["feat"]
    test_node_labels = test_patch_tree.ndata["labels"]

    model = SAGE(in_feats=train_features_dim, hid_feats=50, out_feats=1)
    opt = th.optim.Adam(model.parameters())
    val_val_list = []
    epoch_num = 1
    for epoch in range(epoch_num):
        print('Epoch {}'.format(epoch))
        model.train()
        logits = model(train_patch_tree, train_node_features)
        loss = F.mse_loss(logits.squeeze(-1), train_node_labels)
        r2_lost_train = R2_score(logits.squeeze(-1), train_node_labels)
        print('r2_lost_train = {:.4f}'.format(r2_lost_train.item()))

        opt.zero_grad()
        loss.backward()
        opt.step()

        val_r2_loss = evaluate(model, test_patch_tree, test_node_features, test_node_labels)
        print('val_r2_loss = {:.4f}'.format(val_r2_loss.item()))

        val_val_list.append(val_r2_loss.item())
        print('loss = {:.4f}'.format(loss.item()))

    x_axis = range(1, epoch_num+1, 1)
    plt.plot(x_axis, val_val_list, color='blue', label='val_val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('R2')
    plt.show()


def gnn_test_pos(train_patch_tree: dgl, test_patch_tree):
    train_features_dim = train_patch_tree.ndata["feat"].shape[1]
    train_node_features = train_patch_tree.ndata["feat"]
    train_node_labels = train_patch_tree.ndata["labels"]

    test_node_features = test_patch_tree.ndata["feat"]
    test_node_labels = test_patch_tree.ndata["labels"]

    model = SAGE(in_feats=train_features_dim, hid_feats=100, out_feats=1)
    opt = th.optim.Adam(model.parameters())

    val_val_list = []
    epoch_num = 50
    for epoch in range(epoch_num):
        print('Epoch {}'.format(epoch))
        model.train()
        logits = model(train_patch_tree, train_node_features)
        loss = F.mse_loss(logits.squeeze(-1), train_node_labels)

        opt.zero_grad()
        loss.backward()
        opt.step()
        print('loss = {:.4f}'.format(loss.item()))

        val_r2_loss = evaluate(model, test_patch_tree, test_node_features, test_node_labels)
        print('val_r2_loss = {:.4f}'.format(val_r2_loss.item()))

        val_val_list.append(val_r2_loss.item())

    # node_range = [50,100, 250, 500, 1000, 2500, 5000, 8000, 10000]
    node_range = [50,100]

    node_num_list = []
    eval_position_list = []
    for node_num in node_range:
        print("node_range:", node_num)
        for i in range(100):
            print("position test tree:", i)
            test_tree = test_data_process(1, node_num+1*i)
            node_features = test_tree.ndata["feat"]
            node_labels = test_tree.ndata["labels"]
            eval_position = evaluate_position(model, test_tree, node_features, node_labels)
            node_num_list.append(node_num+1*i)
            eval_position_list.append(eval_position)
    dataframe = pd.DataFrame({'node_num_list': node_num_list, 'eval_position_list': eval_position_list})
    dataframe.to_csv("eval_position_ER_new.csv", index=False, sep=',')


def gnn_top_k_overlap(train_patch_tree: dgl, test_patch_tree):
    train_features_dim = train_patch_tree.ndata["feat"].shape[1]
    train_node_features = train_patch_tree.ndata["feat"]
    train_node_labels = train_patch_tree.ndata["labels"]

    test_node_features = test_patch_tree.ndata["feat"]
    test_node_labels = test_patch_tree.ndata["labels"]

    model = SAGE(in_feats=train_features_dim, hid_feats=100, out_feats=1)
    opt = th.optim.Adam(model.parameters())

    val_val_list = []
    epoch_num = 50
    for epoch in range(epoch_num):
        print('Epoch {}'.format(epoch))
        model.train()
        logits = model(train_patch_tree, train_node_features)
        loss = F.mse_loss(logits.squeeze(-1), train_node_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('loss = {:.4f}'.format(loss.item()))

        val_r2_loss = evaluate(model, test_patch_tree, test_node_features, test_node_labels)
        print('val_r2_loss = {:.4f}'.format(val_r2_loss.item()))

        val_val_list.append(val_r2_loss.item())

    # node_range = [50,100, 250, 500, 1000, 2500]
    node_range = [50,100]
    for node_num in node_range:
        print("node_range:", node_num)
        node_num_list = []
        all_real_labels_list = []
        all_eval_label_list = []
        for i in range(100):
            print("position test tree:", i)
            test_tree = test_data_process(1, node_num+1*i)
            node_features = test_tree.ndata["feat"]
            node_labels = test_tree.ndata["labels"]
            real_labels_list, eval_label_list = evaluate_prob(model, test_tree, node_features, node_labels)
            all_real_labels_list.append(real_labels_list)
            all_eval_label_list.append(eval_label_list)
            node_num_list.append(node_num+1*i)

        dataframe = pd.DataFrame({'node_num_list': node_num_list, 'all_real_labels_list': all_real_labels_list, 'all_eval_label_list': all_eval_label_list})
        dataframe.to_csv("label_list/label_list_SM_"+str(node_num)+".csv", index=False, sep=',')


def real_data_process(file_path: str):
    data = pd.read_csv(file_path, header=None)
    edges = [tuple(x) for x in data.values]
    tree = nx.Graph()
    tree.add_edges_from(edges)
    
    largest_cc = max(nx.connected_components(tree), key=len)
    tree = tree.subgraph(largest_cc).copy()
    tree = nx.convert_node_labels_to_integers(tree)
    
    dgl_tree = data_process_for_single_tree(tree)
    return dgl_tree

def gnn_test_real_networks(train_patch_tree):
    train_features_dim = train_patch_tree.ndata["feat"].shape[1]
    train_node_features = train_patch_tree.ndata["feat"]
    train_node_labels = train_patch_tree.ndata["labels"]

    model = SAGE(in_feats=train_features_dim, hid_feats=100, out_feats=1)
    opt = th.optim.Adam(model.parameters())

    print("Training model on simulated trees to test on real datasets...")
    epoch_num = 50
    for epoch in range(epoch_num):
        model.train()
        logits = model(train_patch_tree, train_node_features)
        loss = F.mse_loss(logits.squeeze(-1), train_node_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: loss = {loss.item():.4f}')

    real_files = [
        "data/hk_csv/bfs_part_tree_10.csv",
        "data/hk_csv/dfs_part_tree_10.csv",
        "data/tw_csv/tw_data_43_tree.csv",
        "data/tw_csv/tw_data_76_tree.csv"
    ]
    
    results = []
    print("\nEvaluating on real-world datasets:")
    for f in real_files:
        test_tree = real_data_process(f)
        node_features = test_tree.ndata["feat"]
        node_labels = test_tree.ndata["labels"]
        eval_position = evaluate_position(model, test_tree, node_features, node_labels)
        print(f"Predicted position for {f}: Rank {eval_position}")
        results.append((f, eval_position))
        
    df = pd.DataFrame(results, columns=["Dataset", "Predicted_Rank"])
    df.to_csv("eval_position_real_data.csv", index=False)
    print("Real data evaluation saved to eval_position_real_data.csv")


# ---------------------------------------------------------------------------
# Ablation study: original (8 feat) vs enriched (10 feat)  — DeepTrace++
# ---------------------------------------------------------------------------

def run_ablation_study(tree_num_train: int = 20, node_num_train: int = 150,
                       test_tree_num: int = 5, test_node_num: int = 100,
                       epoch_num: int = 50, runs: int = 3):
    """
    Train and evaluate two SAGE models side-by-side:
      - 'original'  : 8-feature vector (standard DeepTrace)
      - 'enriched'  : 10-feature vector (DeepTrace++ with closeness
                       centrality and normalised distance-to-index)

    Logged metrics per run:
      - final training loss
      - final validation R²
      - mean predicted position (rank of the true superspreader across
        100 test trees with node_num = 100)

    Results are printed to stdout and saved to ablation_study.csv.

    Parameters
    ----------
    tree_num_train  : trees per batch for training
    node_num_train  : base node count for training trees
    test_tree_num   : number of test trees for R² evaluation
    test_node_num   : node count for test trees
    epoch_num       : training epochs
    runs            : how many independent runs to average over
    """
    import time

    print("\n" + "=" * 65)
    print("  Ablation Study: Original (8 feat) vs Enriched (10 feat)")
    print("=" * 65)

    records = []

    for enriched in [False, True]:
        feat_label = "enriched (10)" if enriched else "original (8)"
        feat_dim   = ENRICHED_FEAT_DIM if enriched else ORIGINAL_FEAT_DIM

        run_losses, run_r2s, run_positions = [], [], []
        t_start = time.time()

        for run_idx in range(runs):
            # ── Build training batch ──────────────────────────────────
            all_train = []
            for i in range(1, tree_num_train + 1):
                patch = train_data_process(2, node_num_train + i, enriched=enriched)
                all_train.append(patch)
            train_batch = dgl.batch(all_train)
            test_batch  = test_data_process(test_tree_num, test_node_num,
                                            enriched=enriched)

            train_feats  = train_batch.ndata["feat"]
            train_labels = train_batch.ndata["labels"]
            test_feats   = test_batch.ndata["feat"]
            test_labels  = test_batch.ndata["labels"]

            # ── Build model matching feature dimension ────────────────
            model = SAGE(in_feats=feat_dim, hid_feats=100, out_feats=1)
            opt   = th.optim.Adam(model.parameters())

            final_loss, final_r2 = 0.0, 0.0
            for epoch in range(epoch_num):
                model.train()
                logits     = model(train_batch, train_feats)
                loss       = F.mse_loss(logits.squeeze(-1), train_labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if epoch == epoch_num - 1:
                    final_loss = loss.item()
                    final_r2   = evaluate(model, test_batch, test_feats,
                                          test_labels).item()

            # ── Position accuracy on 100 test trees ──────────────────
            positions = []
            for j in range(100):
                t = test_data_process(1, test_node_num + j, enriched=enriched)
                pos = evaluate_position(model, t, t.ndata["feat"],
                                        t.ndata["labels"])
                positions.append(pos)
            mean_pos = float(np.mean(positions))

            run_losses.append(final_loss)
            run_r2s.append(final_r2)
            run_positions.append(mean_pos)
            print(f"  [{feat_label}] run {run_idx+1}/{runs} "
                  f"| loss={final_loss:.1f} | R²={final_r2:.4f} "
                  f"| mean_rank={mean_pos:.3f}")

        records.append({
            "features":       feat_label,
            "feat_dim":       feat_dim,
            "runs":           runs,
            "mean_loss":      float(np.mean(run_losses)),
            "std_loss":       float(np.std(run_losses)),
            "mean_val_r2":    float(np.mean(run_r2s)),
            "std_val_r2":     float(np.std(run_r2s)),
            "mean_rank":      float(np.mean(run_positions)),
            "std_rank":       float(np.std(run_positions)),
            "wall_time_s":    round(time.time() - t_start, 2),
        })

    df = pd.DataFrame(records)
    print("\n── Ablation Summary ────────────────────────────────────────")
    print(df[["features", "feat_dim", "mean_loss", "mean_val_r2",
              "mean_rank", "std_rank", "wall_time_s"]].to_string(index=False))
    df.to_csv("ablation_study.csv", index=False)
    print("Full ablation results saved to ablation_study.csv")
    return df


# ---------------------------------------------------------------------------
# Edge-weight comparison  —  DeepTrace++
# ---------------------------------------------------------------------------

def compare_weighted_unweighted(
    tree_num_train: int = 20,
    node_num_train: int = 150,
    test_node_num:  int = 100,
    epoch_num:      int = 50,
    runs:           int = 3,
    weight_modes:   list = None,
) -> pd.DataFrame:
    """
    Train and compare four model variants:

    ┌─────────────────┬────────────────────────────────────────────────┐
    │ Variant         │ Description                                    │
    ├─────────────────┼────────────────────────────────────────────────┤
    │ unweighted      │ Original SAGE (lstm aggregator, no edata)      │
    │ weighted-degree │ WeightedSAGE, w ∝ 1/degree(u), row-softmax     │
    │ weighted-uniform│ WeightedSAGE, uniform weights (SAGEConv mean   │
    │                 │ equivalent — shows pure architecture benefit)  │
    │ weighted-random │ WeightedSAGE, random weights (sanity baseline) │
    └─────────────────┴────────────────────────────────────────────────┘

    Metrics logged per run
    ----------------------
    - final training MSE loss
    - final validation R²
    - mean rank of true superspreader across 100 held-out test trees
    - wall time (seconds)

    Results are printed as a table and saved to edge_weight_comparison.csv.

    Parameters
    ----------
    tree_num_train : trees per training batch
    node_num_train : base node count for training trees
    test_node_num  : node count for test trees
    epoch_num      : training epochs per run
    runs           : independent runs to average over
    weight_modes   : list of weight-mode strings to benchmark
                     (default: ['degree', 'uniform', 'random'])
    """
    if weight_modes is None:
        weight_modes = ['degree', 'uniform', 'random']

    print("\n" + "=" * 65)
    print("  Edge-Weight Comparison (DeepTrace++)")
    print("=" * 65)

    # Build shared training/test graphs once per run (same for all variants)
    records = []

    # ── Unweighted original SAGE ────────────────────────────────────────
    for run_idx in range(runs):
        all_train = [train_data_process(2, node_num_train + i)
                     for i in range(1, tree_num_train + 1)]
        train_batch  = dgl.batch(all_train)
        test_batch   = test_data_process(5, test_node_num)
        feat_dim     = train_batch.ndata["feat"].shape[1]
        train_feats  = train_batch.ndata["feat"]
        train_labels = train_batch.ndata["labels"]
        test_feats   = test_batch.ndata["feat"]
        test_labels  = test_batch.ndata["labels"]

        model = SAGE(in_feats=feat_dim, hid_feats=100, out_feats=1)
        opt   = th.optim.Adam(model.parameters())
        t0    = time.time()

        final_loss = final_r2 = 0.0
        for epoch in range(epoch_num):
            model.train()
            logits = model(train_batch, train_feats)
            loss   = F.mse_loss(logits.squeeze(-1), train_labels)
            opt.zero_grad(); loss.backward(); opt.step()
            if epoch == epoch_num - 1:
                final_loss = loss.item()
                final_r2   = evaluate(model, test_batch, test_feats,
                                      test_labels).item()

        positions = []
        for j in range(100):
            t = test_data_process(1, test_node_num + j)
            positions.append(evaluate_position(
                model, t, t.ndata["feat"], t.ndata["labels"]))
        mean_pos = float(np.mean(positions))

        print(f"  [unweighted]      run {run_idx+1}/{runs} "
              f"| loss={final_loss:.1f} | R²={final_r2:.4f} "
              f"| mean_rank={mean_pos:.3f} | {time.time()-t0:.1f}s")
        records.append({
            "variant": "unweighted", "weight_mode": "-", "run": run_idx+1,
            "loss": final_loss, "val_r2": final_r2,
            "mean_rank": mean_pos, "wall_time_s": round(time.time()-t0, 2),
        })

    # ── WeightedSAGE variants ───────────────────────────────────────────
    for w_mode in weight_modes:
        for run_idx in range(runs):
            all_train = [train_data_process(2, node_num_train + i)
                         for i in range(1, tree_num_train + 1)]
            train_batch  = dgl.batch(all_train)
            test_batch   = test_data_process(5, test_node_num)
            feat_dim     = train_batch.ndata["feat"].shape[1]

            # Assign weights once; they persist on the batched graph
            assign_edge_weights(train_batch, mode=w_mode)
            assign_edge_weights(test_batch,  mode=w_mode)

            train_feats  = train_batch.ndata["feat"]
            train_labels = train_batch.ndata["labels"]
            test_feats   = test_batch.ndata["feat"]
            test_labels  = test_batch.ndata["labels"]

            model = WeightedSAGE(in_feats=feat_dim, hid_feats=100, out_feats=1)
            opt   = th.optim.Adam(model.parameters())
            t0    = time.time()

            final_loss = final_r2 = 0.0
            for epoch in range(epoch_num):
                model.train()
                logits = model(train_batch, train_feats)
                loss   = F.mse_loss(logits.squeeze(-1), train_labels)
                opt.zero_grad(); loss.backward(); opt.step()
                if epoch == epoch_num - 1:
                    final_loss = loss.item()
                    # Re-assign weights to fresh test batch for eval
                    assign_edge_weights(test_batch, mode=w_mode)
                    final_r2 = evaluate(model, test_batch, test_feats,
                                        test_labels).item()

            positions = []
            for j in range(100):
                t = test_data_process(1, test_node_num + j)
                assign_edge_weights(t, mode=w_mode)
                positions.append(evaluate_position(
                    model, t, t.ndata["feat"], t.ndata["labels"]))
            mean_pos = float(np.mean(positions))

            variant = f"weighted-{w_mode}"
            print(f"  [{variant:17s}] run {run_idx+1}/{runs} "
                  f"| loss={final_loss:.1f} | R²={final_r2:.4f} "
                  f"| mean_rank={mean_pos:.3f} | {time.time()-t0:.1f}s")
            records.append({
                "variant": variant, "weight_mode": w_mode, "run": run_idx+1,
                "loss": final_loss, "val_r2": final_r2,
                "mean_rank": mean_pos, "wall_time_s": round(time.time()-t0, 2),
            })

    # ── Summary (averaged over runs) ────────────────────────────────────
    df_full = pd.DataFrame(records)
    summary = (df_full.groupby("variant")
               .agg(mean_loss=("loss", "mean"),
                    std_loss=("loss", "std"),
                    mean_val_r2=("val_r2", "mean"),
                    mean_rank=("mean_rank", "mean"),
                    std_rank=("mean_rank", "std"),
                    mean_time=("wall_time_s", "mean"))
               .reset_index())

    print("\n── Edge-Weight Comparison Summary ──────────────────────────")
    print(summary[["variant", "mean_loss", "mean_val_r2",
                   "mean_rank", "std_rank", "mean_time"]].to_string(index=False))
    df_full.to_csv("edge_weight_comparison.csv", index=False)
    print("Full results saved to edge_weight_comparison.csv")
    return summary


if __name__ == '__main__':
    all_train_tree_list = []
    for i in range(1, 100, 1):
        print("construct tree:", i)
        train_patch_tree = train_data_process(2, 200+i)
        all_train_tree_list.append(train_patch_tree)
    all_train_tree = dgl.batch(all_train_tree_list)
    test_patch_tree = test_data_process(5, 100)

    gnn_test_pos(all_train_tree, test_patch_tree)
    gnn_test_real_networks(all_train_tree)

    # ── DeepTrace++ ablation (toggle enriched features on/off) ──────────
    run_ablation_study(tree_num_train=20, node_num_train=150,
                       epoch_num=50, runs=3)

    # ── DeepTrace++ ablation: edge weights ───────────────────────────────
    compare_weighted_unweighted(tree_num_train=20, node_num_train=150,
                                epoch_num=50, runs=3)
    # gnn_top_k_overlap(all_train_tree, test_patch_tree)