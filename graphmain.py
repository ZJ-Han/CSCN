import argparse
import torch
import numpy as np
from utils import ensure_tu_dataset, load_data
from rankbuild import compute_pagerank_list
from train import cross_validate

def main():
    parser = argparse.ArgumentParser("CSCN Single Run")
    parser.add_argument("--dataset", type=str, default="MUTAG")
    parser.add_argument("--use_node_labels", action='store_true', default=True)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--numofary", type=int, default=3)
    parser.add_argument("--channels", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--k_nodes", type=int, nargs="+", default=[3, 3])
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 1])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--folds", type=int, default=1)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_name = ensure_tu_dataset(args.dataset, root="datasets")
    try:
        adj_list, feat_list, graph_labels = load_data(ds_name, use_node_labels=args.use_node_labels)
    except Exception as e:
        print(f"Error: {e}")
        return
    graph_labels = np.asarray(graph_labels, dtype=np.int64)
    u_labels = np.unique(graph_labels)
    label_map = {l: i for i, l in enumerate(u_labels)}
    graph_labels = np.array([label_map[l] for l in graph_labels])
    pr_list = compute_pagerank_list(adj_list)
    channels = args.channels
    k_nodes = args.k_nodes
    strides = args.strides
    if len(channels) > args.layers - 1:
        channels = channels[:args.layers - 1]
        k_nodes = k_nodes[:args.layers - 1]
        strides = strides[:args.layers - 1]
    while len(channels) < args.layers - 1:
        channels.append(channels[-1])
        k_nodes.append(k_nodes[-1])
        strides.append(strides[-1])

    print(f"Running CSCN on {ds_name} (K={args.layers}, m={args.numofary})")
    acc, std = cross_validate(
        adj_list, feat_list, pr_list, graph_labels,
        K=args.layers, m=args.numofary,
        channels=channels, k_nodes=k_nodes, strides=strides,
        epochs=args.epochs, batch_size=args.batch_size,
        device=device, lr=args.lr, seed=args.seed, folds=args.folds, verbose=True
    )
    print(f"Final Result: {acc:.4f}")

if __name__ == "__main__":
    main()