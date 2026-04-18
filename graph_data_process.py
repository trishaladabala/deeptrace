import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from math import floor
from networkx.algorithms.approximation.steinertree import metric_closure
from math import log
import copy

# ---------------------------------------------------------------------------
# Feature dimension constants — update these if you change feature sets
# ---------------------------------------------------------------------------
ORIGINAL_FEAT_DIM = 8   # original DeepTrace features
ENRICHED_FEAT_DIM = 10  # +2: closeness_centrality, norm_distance_to_index
TEMPORAL_FEAT_DIM = 11  # +3: discovery_order, time_step, growth_stage

class NFeatureDict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


class TreeDataProcess(object):
    def __init__(self, tree: nx.Graph):
        self.tree = tree
        self.degree_list = nx.degree(tree)
        self.degree_dict = dict(list(nx.degree(self.tree)))
        self.all_degree_num = nx.number_of_edges(tree)
        self.all_node_num = nx.number_of_nodes(tree)
        self.layer_num = 0

    def layer_rate_cal(self):
        node_idx_list = list(range(self.all_node_num))
        inital_val = [0]*self.all_node_num
        layer_dict = dict(zip(node_idx_list, inital_val))
        degree_dict = copy.deepcopy(self.degree_dict)
        i = 0
        while len(degree_dict) > 0:
            i += 1
            leaf_node_list = [k for k, v in degree_dict.items() if v <= 1]
            for leaf_node in leaf_node_list:
                del degree_dict[leaf_node]
                layer_dict[leaf_node] = i
                leaf_neighbor = self.tree.neighbors(leaf_node)
                for m in leaf_neighbor:
                    if m in degree_dict.keys():
                        degree_dict[m] -= 1
        layer_value = list(layer_dict.values())
        layer_rate_list = np.array(layer_value)/max(layer_dict.values())
        layer_dict = dict(zip(node_idx_list, layer_rate_list))
        self.layer_num = i

        return layer_dict

    def get_uninfected_node_list(self):
        leafed_index_list = [i[0] for i in self.degree_list if i[1] == 1]
        random.seed(10)
        uninft_size = len(leafed_index_list)
        uninft_node_list = random.sample(leafed_index_list, floor(0.6 * uninft_size))
        return uninft_node_list

    def nfeature_process(self, enriched: bool = False):
        """
        Compute per-node features for the GNN.

        Original 8 features (enriched=False):
            node_num, degree_per, degree_per_aver,
            inft_ndegree_per, inft_alldegree_per,
            distance_per, layer_rate, layer_num

        Enriched 10 features (enriched=True)  — DeepTrace++:
            above 8  +  closeness_centrality  +  norm_dist_to_index

        Parameters
        ----------
        enriched : bool
            If True, append closeness centrality and normalised
            shortest-path distance from the highest-degree node
            (used as a proxy for the index case when the true
            source is unknown).
        """
        avg_degree = self.all_degree_num / self.all_node_num
        nfeature_dict = NFeatureDict()
        for k, v in self.degree_dict.items():
            nfeature_dict[k]["node_num"] = log(self.all_node_num)
            nfeature_dict[k]["degree_per"] = v / self.all_degree_num
            nfeature_dict[k]["degree_per_aver"] = v / avg_degree

        uninft_node_list = self.get_uninfected_node_list()
        inft_node_size = self.all_node_num - len(uninft_node_list)

        tree_adj_dict = self.tree.adj.copy()
        for k, v in tree_adj_dict.items():
            infected_adj_k_list = list(set(v.keys()).difference(set(uninft_node_list)))
            infected_adj_k_size = len(infected_adj_k_list)
            nfeature_dict[k]["inft_ndegree_per"] = infected_adj_k_size / self.degree_dict[k]
            nfeature_dict[k]["inft_alldegree_per"] = infected_adj_k_size / inft_node_size

        closure_graph = metric_closure(self.tree, weight='weight')
        closure_graph_deg = closure_graph.degree(weight='distance')
        closure_graph_size = closure_graph.size(weight='distance')
        for i in closure_graph_deg:
            nfeature_dict[i[0]]["distance_per"] = i[1] / closure_graph_size

        layer_dict_temp = self.layer_rate_cal()
        for k, v in layer_dict_temp.items():
            nfeature_dict[k]["layer_rate"] = float(v)
            nfeature_dict[k]["layer_num"] = log(self.layer_num)

        # ── DeepTrace++ enrichment ────────────────────────────────────────
        if enriched:
            nfeature_dict = self._enrich_features(nfeature_dict)

        return nfeature_dict

    # ------------------------------------------------------------------
    def _enrich_features(self, nfeature_dict: 'NFeatureDict') -> 'NFeatureDict':
        """
        Append two centrality-aware features to every node entry.

        Feature 9  — closeness_centrality
            C(v) = (n-1) / sum_u d(v,u)
            Measures how close a node is to all others.  A true epidemic
            source tends to sit near the centroid of the infection tree,
            giving it a high closeness score.
            Normalised by max centrality in the graph so values ∈ [0, 1].

        Feature 10 — norm_dist_to_index
            d(v, v_index) / diameter(G)
            Shortest-path hop distance from each node to the *index node*
            (highest-degree node, our proxy for the first known case).
            Normalised by graph diameter so values ∈ [0, 1].
            Nodes close to the index case receive low scores, which
            distinguishes peripheral spreaders from the core cluster.
        """
        # ── Closeness centrality ──────────────────────────────────────
        closeness = nx.closeness_centrality(self.tree)
        max_closeness = max(closeness.values()) if closeness else 1.0
        if max_closeness == 0:
            max_closeness = 1.0

        # ── Index node: highest-degree node as proxy for first case ──
        index_node   = max(self.degree_dict, key=lambda n: self.degree_dict[n])
        path_lengths = nx.single_source_shortest_path_length(self.tree, index_node)

        # Diameter — use eccentricity for speed; fallback to max path len
        try:
            diam = nx.diameter(self.tree)
        except nx.NetworkXError:
            diam = max(path_lengths.values()) if path_lengths else 1
        diam = diam if diam > 0 else 1

        for node in self.degree_dict:
            nfeature_dict[node]["closeness_centrality"] = (
                closeness.get(node, 0.0) / max_closeness
            )
            nfeature_dict[node]["norm_dist_to_index"] = (
                path_lengths.get(node, diam) / diam
            )

        return nfeature_dict


if __name__ == '__main__':
    ER = nx.random_graphs.barabasi_albert_graph(50, 2)
    tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
    # tree_ka.remove_node(1)
    graph1 = TreeDataProcess(tree_ka)
    graph1.nfeature_process()
    # graph1.layer_rate_cal()

    # patch_tree = to_networkx(graph1)
    nx.draw(tree_ka, node_size=100, with_labels=True)
    plt.show()

