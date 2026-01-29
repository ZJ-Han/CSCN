import os
import urllib.request
import zipfile
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

def ensure_tu_dataset(name: str, root: str = "datasets"):
    name = name.upper()
    ds_dir = os.path.join(root, name)
    indicator_path = os.path.join(ds_dir, f"{name}_graph_indicator.txt")
    if os.path.exists(indicator_path):
        return name
    os.makedirs(ds_dir, exist_ok=True)
    url = f"http://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{name}.zip"
    zip_path = os.path.join(ds_dir, f"{name}.zip")
    print(f"Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(ds_dir)
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
    return name

def load_data(ds_name, use_node_labels):
    path = f"datasets/{ds_name}"
    graph_indicator = np.loadtxt(f"{path}/{ds_name}_graph_indicator.txt", dtype=np.int64)
    _, graph_size = np.unique(graph_indicator, return_counts=True)
    edges = np.loadtxt(f"{path}/{ds_name}_A.txt", dtype=np.int64, delimiter=",")
    edges -= 1
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                   shape=(graph_indicator.size, graph_indicator.size))
    A = A.maximum(A.T)
    A.setdiag(0)
    if use_node_labels:
        x = np.loadtxt(f"{path}/{ds_name}_node_labels.txt", dtype=np.int64).reshape(-1, 1)
        enc = OneHotEncoder(sparse=False)
        x = enc.fit_transform(x)
    else:
        deg = A.sum(axis=1).A1.astype(np.int64)
        max_deg = deg.max()
        enc = OneHotEncoder(categories=[np.arange(max_deg + 1)], sparse=False)
        x = enc.fit_transform(deg.reshape(-1, 1))
    adj = []
    features = []
    idx = 0
    for i in range(graph_size.size):
        n_i = graph_size[i]
        adj.append(A[idx:idx + n_i, idx:idx + n_i])
        features.append(x[idx:idx + n_i, :])
        idx += n_i
    class_labels = np.loadtxt(f"{path}/{ds_name}_graph_labels.txt", dtype=np.int64)
    return adj, features, class_labels