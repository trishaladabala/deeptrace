import os
import dgl
import torch as th
import torch.nn.functional as F
import networkx as nx
import pandas as pd
import numpy as np
import time

from model import SAGE, WeightedSAGE, assign_edge_weights
from graph_data_process import TreeDataProcess, ORIGINAL_FEAT_DIM, ENRICHED_FEAT_DIM
from cal_max_min_ds import CalMaxMinDS
from evaluation import build_training_data, eval_topk_accuracy, eval_hop_error
import topk_tracing


def real_data_process(file_path: str, enriched: bool = False):
    """
    Load real-world contact tracing tree, convert to DGL, and find ground truths.
    """
    data = pd.read_csv(file_path, header=None)
    edges = [tuple(x) for x in data.values]
    tree = nx.Graph()
    tree.add_edges_from(edges)
    

    largest_cc = max(nx.connected_components(tree), key=len)
    tree = tree.subgraph(largest_cc).copy()
    

    tree = nx.convert_node_labels_to_integers(tree)
    
    proc = TreeDataProcess(tree)
    unift = proc.get_uninfected_node_list()
    nfeat = proc.nfeature_process(enriched=enriched)
    tree.remove_nodes_from(list(unift))
    

    labels = []
    nodes_list = list(nx.nodes(tree))
    for node in nodes_list:
        cal = CalMaxMinDS(tree, unift, node)
        try:
            mx = cal.cal_max_ds()
            mn = cal.cal_min_ds()
            labels.append(np.sqrt(mx * mn))
        except (ZeroDivisionError, ValueError):
            labels.append(0.0)
            
    rumor_center_idx = int(np.argmax(labels))
    

    feat_rows = []
    for node in nodes_list:
        v = nfeat[node]
        row = [v["node_num"], v["degree_per"], v["degree_per_aver"],
               v["inft_ndegree_per"], v["inft_alldegree_per"],
               v["distance_per"], v["layer_rate"], v["layer_num"]]
        if enriched:
            row.append(v["closeness_centrality"])
            row.append(v["norm_dist_to_index"])
        feat_rows.append(row)
        
    g_nfeature = th.tensor(feat_rows, dtype=th.float32)
    dgl_tree = dgl.from_networkx(tree)
    dgl_tree.ndata["feat"] = g_nfeature
    
    return dgl_tree, tree, rumor_center_idx, labels


def run_real_world_evaluation():
    print("======================================================================")
    print("  DeepTrace vs DeepTrace++ — Real-World Dataset Evaluation")
    print("======================================================================")

    print("\n[1/3] Building synthetic training data (ER networks)...")

    train_trees_base = build_training_data(num_trees=30, node_range=(80, 150), enriched=False)
    train_trees_enriched = build_training_data(num_trees=30, node_range=(80, 150), enriched=True)
    
    print("[2/3] Training models...")
    models = {
        "DeepTrace": SAGE(ORIGINAL_FEAT_DIM, 100, 1),
        "DT++ enriched": SAGE(ENRICHED_FEAT_DIM, 100, 1),
        "DT++ weighted": WeightedSAGE(ORIGINAL_FEAT_DIM, 100, 1),
        "DeepTrace++": WeightedSAGE(ENRICHED_FEAT_DIM, 100, 1)
    }
    
    optimizers = {name: th.optim.Adam(m.parameters(), lr=0.01) for name, m in models.items()}
    
    epochs = 100
    for epoch in range(epochs):
        for name, model in models.items():
            model.train()
            is_enriched = "enriched" in name or name == "DeepTrace++"
            is_weighted = "weighted" in name or name == "DeepTrace++"
            
            dgl_t = train_trees_enriched if is_enriched else train_trees_base
            
            if is_weighted:
                assign_edge_weights(dgl_t, mode='degree')
                
            feats = dgl_t.ndata["feat"].float()
            labels = dgl_t.ndata["labels"].float()
            
            logits = model(dgl_t, feats).squeeze(-1)
            loss = F.mse_loss(logits, labels)
            
            opt = optimizers[name]
            opt.zero_grad()
            loss.backward()
            opt.step()
                
        if (epoch+1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{epochs} complete.")

    real_files = [
        "data/hk_csv/bfs_part_tree_10.csv",
        "data/hk_csv/dfs_part_tree_10.csv",
        "data/tw_csv/tw_data_43_tree.csv",
        "data/tw_csv/tw_data_76_tree.csv"
    ]
    
    print("\n[3/3] Evaluating on real-world Hong Kong & Taiwan Datasets...")
    
    results = []
    for f_path in real_files:
        if not os.path.exists(f_path):
            print(f"Skipping {f_path} (File not found)")
            continue
            
        print(f"\n── Dataset: {os.path.basename(f_path)} ──")
        
        dgl_base, tree_nx, true_source, _ = real_data_process(f_path, enriched=False)
        dgl_enriched, _, _, _ = real_data_process(f_path, enriched=True)
        
        for name, model in models.items():
            model.eval()
            is_enriched = "enriched" in name or name == "DeepTrace++"
            is_weighted = "weighted" in name or name == "DeepTrace++"
            
            dgl_t = dgl_enriched if is_enriched else dgl_base
            if is_weighted:
                assign_edge_weights(dgl_t, mode='degree')
                
            start_t = time.time()
            with th.no_grad():
                feats = dgl_t.ndata["feat"].float()
                preds = model(dgl_t, feats).squeeze(-1).numpy()
            
            end_t = time.time()
            
            from evaluation import forward_trace_first_detection
            first_detect, _, _ = forward_trace_first_detection(
                tree=tree_nx,
                true_source=true_source,
                model=model,
                enriched=is_enriched,
                weighted_mode='degree' if is_weighted else None,
                strategy="bfs"
            )
            
            nodes = list(tree_nx.nodes())
            preds_t = th.tensor(preds)
            labels_t = th.tensor(labels)
            
            top_k = eval_topk_accuracy(preds_t, labels_t)
            hop_err = eval_hop_error(tree_nx, preds_t, labels_t, nodes)
            
            results.append({
                "dataset": os.path.basename(f_path),
                "model": name,
                "top1_acc": top_k[1],
                "top5_acc": top_k[5],
                "top10_acc": top_k[10],
                "hop_error": hop_err,
                "first_detect": first_detect,
                "time_s": end_t - start_t
            })
            

    df = pd.DataFrame(results)
    
    print("\n======================================================================")
    print("  REAL-WORLD RESULTS SUMMARY")
    print("======================================================================\n")
    
    for dataset in df['dataset'].unique():
        print(f"── {dataset} ──")
        ds_df = df[df['dataset'] == dataset].drop(columns=['dataset'])
        print(ds_df.to_string(index=False))
        print()
        
    print("── Overall Real-World Average ──")
    agg_df = df.groupby('model').mean(numeric_only=True).reset_index()
    print(agg_df.to_string(index=False))
    
    os.makedirs("evaluation_results", exist_ok=True)
    df.to_csv("evaluation_results/real_world_results.csv", index=False)
    print("\n✓ Results saved to evaluation_results/real_world_results.csv")


if __name__ == "__main__":
    run_real_world_evaluation()
