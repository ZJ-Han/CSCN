import numpy as np
import torch

def pack_tree_sequence_for_node(x_node_feats, tree_of_root, K, m):
    C0 = x_node_feats.shape[1]
    children = tree_of_root["children"]
    positions = [tree_of_root["levels"][0][0]]
    for _ in range(1, K):
        next_pos = []
        for parent in positions:
            next_pos.append(parent)
            ch = children.get(parent, []) if parent is not None else []
            need = max(0, m - len(ch))
            next_pos.extend(ch[:m] + [None] * need)
        positions = next_pos
    d = np.zeros((len(positions), C0), dtype=np.float32)
    for i, nid in enumerate(positions):
        if nid is not None:
            d[i, :] = x_node_feats[nid]
    return d.reshape(-1)

def make_seq_features_list(X_concat_lst, trees_per_graph, K, m):
    seq_feats_lst = []
    for g_idx, Xg in enumerate(X_concat_lst):
        Tg = trees_per_graph[g_idx]
        n_g, C0 = Xg.shape
        F0 = C0 * ((1 + m) ** (K - 1))
        M = np.zeros((n_g, F0), dtype=np.float32)
        for root in range(n_g):
            M[root] = pack_tree_sequence_for_node(Xg, Tg[root], K, m)
        seq_feats_lst.append(M)
    return seq_feats_lst

def prepare_joint_conv_data(feat_list, pr_list, trees_list, K, m, k_node_min=1):
    raw_feats_list = make_seq_features_list(feat_list, trees_list, K=K, m=m)
    processed_feats = []
    graph_ids = []
    R = m + 1
    L_total = (m + 1) ** (K - 1)
    if L_total % R != 0:
        raise ValueError(f"Sequence length error")
    P = L_total // R
    if not feat_list:
        return torch.tensor([]), torch.tensor([])
    C = feat_list[0].shape[1]
    for g_idx, (feats, pr) in enumerate(zip(raw_feats_list, pr_list)):
        n_g = feats.shape[0]
        if n_g == 0:
            continue
        sort_idx = np.argsort(-pr)
        feats_sorted = feats[sort_idx]
        if n_g < k_node_min:
            pad_size = k_node_min - n_g
            padding = np.zeros((pad_size, feats_sorted.shape[1]), dtype=feats_sorted.dtype)
            feats_sorted = np.vstack([feats_sorted, padding])
            n_g = k_node_min
        feats_reshaped = feats_sorted.reshape(n_g, P * R, C)
        feats_reshaped = np.transpose(feats_reshaped, (0, 2, 1))
        feats_reshaped = feats_reshaped.reshape(n_g, C, P, R)
        processed_feats.append(feats_reshaped)
        ids = np.full(n_g, g_idx, dtype=np.int64)
        graph_ids.append(ids)
    if not processed_feats:
        return torch.tensor([]), torch.tensor([])
    X_final = np.vstack(processed_feats)
    ids_final = np.concatenate(graph_ids)
    return torch.from_numpy(X_final).float(), torch.from_numpy(ids_final).long()