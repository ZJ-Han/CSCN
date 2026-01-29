import torch
import torch.nn as nn
import numpy as np
import math
from rankbuild import construct_mary_trees_batch
from packtree import prepare_joint_conv_data
from model import CSCN
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR


def run_epoch_batched(model, optimizer, criterion, X_all, ids_all, y_all,
                      graph_indices, batch_size, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
    total_correct = 0
    total_graphs = len(graph_indices)
    if is_train:
        perm = torch.randperm(total_graphs)
        graph_indices = graph_indices[perm]
    num_batches = math.ceil(total_graphs / batch_size)
    max_id = ids_all.max().item()
    mapping_tensor = torch.full((max_id + 1,), -1, dtype=torch.long, device=device)
    with torch.set_grad_enabled(is_train):
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, total_graphs)
            batch_gids = graph_indices[start:end]
            curr_batch_size = len(batch_gids)
            node_mask = torch.isin(ids_all, batch_gids)
            X_batch = X_all[node_mask]
            id_batch_raw = ids_all[node_mask]
            y_batch = y_all[batch_gids]
            if X_batch.size(0) == 0: continue
            mapping_tensor[batch_gids] = torch.arange(curr_batch_size, device=device)
            id_batch_remapped = mapping_tensor[id_batch_raw]
            if is_train: optimizer.zero_grad()
            out_batch = model(X_batch, id_batch_remapped)
            loss = criterion(out_batch, y_batch)
            if is_train:
                loss.backward()
                optimizer.step()
            preds = out_batch.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            mapping_tensor[batch_gids] = -1
    return total_correct / total_graphs


def cross_validate(adj_list, feat_list, pr_list, labels, K, m, channels, k_nodes, strides,
                   epochs, batch_size, device, lr=0.001, seed=1, folds=10, verbose=False):
    trees_list = construct_mary_trees_batch(adj_list, m=m, layers=K)
    pad_min = k_nodes[0] if k_nodes else 1
    X_all, ids_all = prepare_joint_conv_data(feat_list, pr_list, trees_list, K, m, k_node_min=pad_min)
    if X_all.size(0) == 0: return 0.0, 0.0
    X_all = X_all.to(device)
    ids_all = ids_all.to(device)
    y_all = torch.tensor(labels, dtype=torch.long).to(device)
    num_graphs = len(labels)
    all_graph_indices = torch.arange(num_graphs, device=device)

    epoch_accs = {ep: [] for ep in range(epochs)}

    if folds <= 1:
        train_idx, test_idx = train_test_split(np.arange(num_graphs), test_size=0.1, random_state=seed, stratify=labels)
        splits = [(train_idx, test_idx)]
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(skf.split(np.zeros(num_graphs), labels))

    for fold, (train_idx, test_idx) in enumerate(splits):
        feat_dim = X_all.shape[1]
        num_classes = len(np.unique(labels))
        model = CSCN(num_classes=num_classes, feat_dim=feat_dim, layers=K, m=m,
                     channels=channels, k_nodes=k_nodes, strides=strides).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        train_gids = all_graph_indices[train_idx]
        test_gids = all_graph_indices[test_idx]

        for ep in range(epochs):
            run_epoch_batched(model, optimizer, criterion, X_all, ids_all, y_all,
                              train_gids, batch_size, device, is_train=True)
            scheduler.step()
            val_acc = run_epoch_batched(model, optimizer, criterion, X_all, ids_all, y_all,
                                        test_gids, batch_size, device, is_train=False)
            epoch_accs[ep].append(val_acc)

            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"Fold {fold + 1} | Epoch {ep + 1} | Acc: {val_acc:.4f}")

        del model
        torch.cuda.empty_cache()

    avg_epoch_accs = {ep: np.mean(epoch_accs[ep]) for ep in range(epochs)}
    best_epoch = max(avg_epoch_accs, key=avg_epoch_accs.get)
    best_epoch_acc = avg_epoch_accs[best_epoch]
    std = np.std(epoch_accs[best_epoch])

    return best_epoch_acc, std