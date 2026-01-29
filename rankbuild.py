import numpy as np
from scipy.sparse import csr_matrix

def compute_pagerank_list(adj_list):
    pr_list = []
    for A in adj_list:
        pr = _pagerank_csr(A)
        pr_list.append(pr)
    return pr_list

def _pagerank_csr(A, alpha=0.85, tol=1e-8, max_iter=200):
    A_sym = A.maximum(A.T).tocsr()
    Acoo = A_sym.tocoo()
    n = A_sym.shape[0]
    deg = np.asarray(A_sym.sum(axis=1)).ravel()
    data = Acoo.data.astype(np.float64).copy()
    rows = Acoo.row
    cols = Acoo.col
    data = data / np.maximum(deg[rows], 1.0)
    P = csr_matrix((data, (rows, cols)), shape=A_sym.shape)
    x = np.full(n, 1.0 / n, dtype=np.float64)
    tele = (1.0 - alpha) / n
    for _ in range(max_iter):
        x_new = alpha * (P.T @ x) + tele
        if np.linalg.norm(x_new - x, 1) < tol:
            x = x_new
            break
        x = x_new
    return x

def _neighbors_1hop(A, u):
    A = A.tocsr()
    return A.indices[A.indptr[u]:A.indptr[u + 1]]

def _khop_nodes(A, root, k):
    n = A.shape[0]
    dist = np.full(n, -1, dtype=np.int32)
    q = [root]
    dist[root] = 0
    for _ in range(k):
        nxt = []
        for u in q:
            for v in _neighbors_1hop(A, u):
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        q = nxt
        if not q:
            break
    return np.where(dist == k)[0]

def _select_children_pagerank(A, pr, parent, m, exclude=None):
    if exclude is None:
        exclude = set()
    chosen = []
    hop1 = [v for v in _neighbors_1hop(A, parent) if v != parent and v not in exclude]
    hop1.sort(key=lambda u: (-pr[u], u))
    if len(hop1) >= m:
        return hop1[:m]
    chosen.extend(hop1)
    used = set(chosen) | exclude | {parent}
    k = 2
    while len(chosen) < m:
        kh = [v for v in _khop_nodes(A, parent, k) if v not in used]
        if not kh:
            break
        kh.sort(key=lambda u: (-pr[u], u))
        for v in kh:
            if len(chosen) >= m:
                break
            chosen.append(v)
            used.add(v)
        k += 1
    return chosen[:m]

def construct_mary_trees_batch(adj_lst, m, layers):
    out = []
    for A in adj_lst:
        A = A.tocsr()
        pr = _pagerank_csr(A)
        roots = list(range(A.shape[0]))
        trees = {}
        for r in roots:
            children = {}
            levels = [[r]]
            parent_of = {r: None}
            frontier = [r]
            for _depth in range(1, layers):
                next_frontier = []
                for u in frontier:
                    excl = {u}
                    if parent_of[u] is not None:
                        excl.add(parent_of[u])
                    ch = _select_children_pagerank(A, pr, u, m, exclude=excl)
                    children[u] = ch
                    for v in ch:
                        parent_of[v] = u
                    next_frontier.extend(ch)
                levels.append(next_frontier.copy())
                frontier = next_frontier
            trees[r] = {"levels": levels, "children": children}
        out.append(trees)
    return out