"""Cache primitives: ε-nets, Reverse Greedy, nearest-neighbour lookups and LFU admission.

Embeddings are L2-normalized, so the distance between two states is ||x - y|| / 2 = sqrt(2 - 2 x·y) / 2,
which lies in [0, 1].
"""
import numpy as np
import pandas as pd

from data import SEED


def distance(A, B, block=2048):
    """Pairwise distances ||a - b|| / 2 between the rows of A and B."""
    A, B = np.asarray(A, np.float32), np.asarray(B, np.float32)
    out = np.empty((len(A), len(B)), dtype=np.float32)
    for s in range(0, len(A), block):
        dots = np.clip(A[s:s + block] @ B.T, -1.0, 1.0)
        out[s:s + block] = np.sqrt(np.maximum(2.0 - 2.0 * dots, 0.0)) / 2.0
    return out


def epsilon_net(X, idx, eps, seed=SEED, max_size=None):
    """Greedy ε-net of the states idx, visited in a seeded random order: a state becomes a
    representative when it is farther than ε from every earlier one. None if more than max_size."""
    idx = np.asarray(idx, dtype=int)
    thr = 1.0 - 2.0 * float(eps) ** 2          # distance > ε  <=>  x·y < thr
    reps = []
    buf, n = np.empty((64, X.shape[1]), dtype=np.float32), 0
    for p in np.random.default_rng(seed).permutation(len(idx)):
        g = int(idx[p])
        x = X[g]
        if n and float(np.max(buf[:n] @ x)) >= thr:
            continue
        if n == len(buf):
            buf = np.concatenate([buf, np.empty_like(buf)])
        buf[n] = x
        n += 1
        reps.append(g)
        if max_size is not None and len(reps) > max_size:
            return None
    return np.asarray(reps, dtype=int)


def assign(X, idx, reps, batch=2048):
    """Cell (nearest representative) of every state in idx and its distance to it."""
    idx = np.asarray(idx, dtype=int)
    R = X[reps]
    cells = np.empty(len(idx), dtype=int)
    dists = np.empty(len(idx), dtype=np.float32)
    for s in range(0, len(idx), batch):
        S = X[idx[s:s + batch]] @ R.T
        j = np.argmax(S, axis=1)
        sim = np.clip(S[np.arange(len(j)), j], -1.0, 1.0)
        cells[s:s + batch] = j
        dists[s:s + batch] = np.sqrt(np.maximum(2.0 - 2.0 * sim, 0.0)) / 2.0
    return cells, dists


def cell_weights(cells, m):
    counts = np.bincount(cells, minlength=m).astype(float)
    return counts / max(counts.sum(), 1.0)


def _two_smallest(sub):
    if sub.shape[1] == 1:
        return (np.zeros(len(sub), int), sub[:, 0].astype(np.float64),
                np.zeros(len(sub), int), np.full(len(sub), np.inf))
    two = np.argpartition(sub, 1, axis=1)[:, :2]
    vals = np.take_along_axis(sub, two, axis=1).astype(np.float64)
    swap = vals[:, 1] < vals[:, 0]
    two[swap] = two[swap][:, ::-1]
    vals[swap] = vals[swap][:, ::-1]
    return two[:, 0], vals[:, 0], two[:, 1], vals[:, 1]


def reverse_greedy(weights, costs, L, k):
    """Start with every column of L in the cache and repeatedly drop the column whose removal
    increases sum_i w_i min{c_i, min_{j in M} L_ij} the least, until k columns are left."""
    w = np.asarray(weights, dtype=np.float64)
    c = np.asarray(costs, dtype=np.float64)
    L = np.asarray(L, dtype=np.float32)
    n_rows, n_cols = L.shape
    if n_cols <= k:
        return np.arange(n_cols)
    best_idx, second_idx = np.empty(n_rows, dtype=int), np.empty(n_rows, dtype=int)
    best_val, second_val = np.empty(n_rows), np.empty(n_rows)
    for s in range(0, n_rows, 2048):
        sl = slice(s, s + 2048)
        best_idx[sl], best_val[sl], second_idx[sl], second_val[sl] = _two_smallest(L[sl])
    active = np.ones(n_cols, dtype=bool)
    for _ in range(n_cols - int(k)):
        base = np.minimum(c, best_val)
        alt = np.minimum(c, second_val)
        inc = np.bincount(best_idx, weights=w * (alt - base), minlength=n_cols)
        inc[~active] = np.inf
        r = int(np.argmin(inc))
        active[r] = False
        aff = np.flatnonzero((best_idx == r) | (second_idx == r))
        if aff.size == 0:
            continue
        act = np.flatnonzero(active)
        bi, bv, si, sv = _two_smallest(L[np.ix_(aff, act)])
        best_idx[aff], best_val[aff] = act[bi], bv
        second_idx[aff], second_val[aff] = act[si], sv
    return np.flatnonzero(active)


def top1(E, cache, idx, batch=1024):
    """Most similar cached state (by cosine of E) of every state in idx, and the similarity."""
    cache = np.asarray(cache, dtype=int)
    C = E[cache]
    cand = np.empty(len(idx), dtype=int)
    sim = np.empty(len(idx), dtype=np.float32)
    for s in range(0, len(idx), batch):
        S = E[idx[s:s + batch]] @ C.T
        j = np.argmax(S, axis=1)
        cand[s:s + batch] = cache[j]
        sim[s:s + batch] = S[np.arange(len(j)), j]
    return cand, sim


def lfu_order(E, conv_ids, pool, batch=1024):
    """LFU admission estimated on the log: the pool states ranked by how often they are the nearest
    neighbour (by E) of a state from another conversation (ties: summed similarity)."""
    pool = np.asarray(pool, dtype=int)
    conv = pd.factorize(pd.Series(conv_ids[pool]))[0]
    Ep = E[pool]
    counts, simsum = np.zeros(len(pool)), np.zeros(len(pool))
    for s in range(0, len(pool), batch):
        S = Ep[s:s + batch] @ Ep.T
        S[conv[s:s + batch][:, None] == conv[None, :]] = -np.inf
        j = np.argmax(S, axis=1)
        counts += np.bincount(j, minlength=len(pool))
        simsum += np.bincount(j, weights=S[np.arange(len(j)), j], minlength=len(pool))
    return np.lexsort((-simsum, -counts))


def lfu_cache(order, pool, k):
    """The k most popular pool states (sorted by index)."""
    return np.asarray(pool, dtype=int)[np.sort(order[: min(int(k), len(pool))])]
