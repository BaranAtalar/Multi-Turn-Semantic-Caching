"""Online experiments (Figure 3b, 3d, Figure 4, Table 1): every method starts with an empty cache and no
labels and serves one stream of all 2,200 conversations (in a seeded random order, turns in order).

* CLCB-LS-MT (Algorithm 2) grows an ε-net of the states it has seen, learns the cost and the reuse risk
  online with kernel ridge regression, and serves each conversation with a snapshot (cache, models)
  taken at the last switch; it also reuses from a session buffer of the conversation's own responses.
  Lower confidence bounds decide: reuse iff c̲(z) > min_u R̲(z, u), with R̲ = min{1, γ̂ (r̂ − β_r σ_r)}.
* Single-turn CUCB-SC-Cont runs the same learner on the request embedding alone with the known
  mismatch γ̂ φ(d) as its risk.
* ElastiCache-MT, ContextCache and the joint-state threshold insert every LLM response and evict the
  least recently used entry; vCache learns a threshold per entry from the judge's verdicts on the
  requests it explores.

The hyperparameters are the offline selections (results/<workload>/offline_loss.csv). Each reuse is
scored with the judge label of its (request, served state) pair; a reuse without a label counts as
unacceptable, and so does a label CLCB-LS-MT asks for that is missing from the table.
"""
import heapq
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, cholesky

from data import SEED
from models import NystromKRR, median_sq_distance
from offline import ALL, BUDGETS, PRICE, VC_FLOOR, vc_alpha, vc_fit, vc_tau

STREAM_SEED = SEED + 101          # order of the conversations
K_GRAPH, K_SERVE = 16, 4          # centers' nearest-neighbour graph; cached centers scored per request
COST_PRIOR, RISK_PRIOR, PRIOR_COUNT = 0.0, 0.5, 1.0
REFRESH_EVERY = 2000              # exact re-inversion of the KRR system after this many rank-one updates
SESSION_MAX_DIST = 0.25           # session candidates must lie within this distance of the request
LANDMARK_SAMPLE = 4               # cost landmarks are drawn from 4 × rank random stream states
BASELINE_BLOCK = 512
VC_ROUND = 16                     # vCache: requests per round of similarity computations
N_BOOT = 2000
ONLINE_BUDGETS = [str(k) for k in BUDGETS] + [ALL]
METHODS = ["CLCB-LS-MT", "ElastiCache-MT", "ContextCache", "vCache", "Joint-state threshold", "Single-turn"]


# ---------------------------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------------------------

class Stream:
    def __init__(self, w):
        order = w.sorted_order                                    # grouped by conversation, turns in order
        cv = w.conv_ids[order]
        starts = np.flatnonzero(np.r_[True, cv[1:] != cv[:-1]])
        groups = np.split(order, starts[1:])
        perm = np.random.default_rng(STREAM_SEED).permutation(len(groups))
        groups = [groups[i] for i in perm]
        self.states = np.concatenate(groups).astype(np.int64)     # state at each stream position
        self.n_conv = len(groups)
        self.bounds = np.r_[0, np.cumsum([len(g) for g in groups])].astype(np.int64)
        self.T = len(self.states)
        self.conv = np.repeat(np.arange(self.n_conv), np.diff(self.bounds))
        self.cost = w.cost[self.states].astype(float)


# ---------------------------------------------------------------------------------------------
# Online kernel ridge regression
# ---------------------------------------------------------------------------------------------

class NystromMap:
    """A fixed Nyström feature map of an RBF kernel, with landmarks drawn once from unlabeled inputs."""

    def __init__(self, pool, rank, gamma, seed=SEED, eig_tol=1e-8):
        pool = np.ascontiguousarray(np.asarray(pool, dtype=np.float32))
        rows = pool.view(np.dtype((np.void, pool.dtype.itemsize * pool.shape[1]))).ravel()
        _, distinct = np.unique(rows, return_index=True)
        pick = np.random.default_rng(seed).permutation(distinct)[: int(rank)]
        self.L = pool[np.sort(pick)].astype(np.float64)
        self.gamma = float(gamma)
        S, U = np.linalg.eigh(NystromKRR.rbf(self.L, self.L, self.gamma))
        keep = S > eig_tol * S.max()
        self.proj = U[:, keep] / np.sqrt(S[keep])
        self.r = int(keep.sum())

    def __call__(self, X, batch=4096):
        X = np.asarray(X, dtype=np.float32)
        out = np.empty((len(X), self.r), dtype=np.float32)
        for s in range(0, len(X), batch):
            out[s:s + batch] = NystromKRR.rbf(X[s:s + batch], self.L, self.gamma) @ self.proj
        return out


class OnlineKRR:
    """The offline Nyström KRR, updated a batch of observations at a time (Woodbury):
    mean(x) = ȳ + z'A⁻¹ Σ z_s (y_s − ȳ), width²(x) = (1 − |z|²)₊ + λ z'A⁻¹z, A = λI + Σ z_s z_s'.
    ȳ starts at a prior mean with a pseudo-count. An input may enter A before its label arrives."""

    def __init__(self, r, ridge, prior_mean, prior_n=1.0):
        self.r, self.lam = int(r), float(ridge)
        self.A = np.eye(self.r) * self.lam
        self.Ainv = np.eye(self.r) / self.lam
        self.b0, self.s1 = np.zeros(self.r), np.zeros(self.r)
        self.ysum, self.nlab, self.nobs = 0.0, 0, 0
        self.prior_mean, self.prior_n = float(prior_mean), float(prior_n)
        self.logdet = 0.0                     # ln det(I + K/λ) over every input so far
        self.pending = {}
        self.since_refresh = 0

    def ybar(self):
        n = self.prior_n + self.nlab
        return self.prior_mean if n <= 0 else (self.prior_mean * self.prior_n + self.ysum) / n

    def theta(self):
        return self.Ainv @ (self.b0 - self.ybar() * self.s1)

    def add_batch(self, Z, y=None, keys=None):
        """Add inputs Z with labels y (or pending, under keys). Returns (U, M⁻¹) for width_drop."""
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        n = len(Z)
        U = self.Ainv @ Z.T
        Lm = cholesky(np.eye(n) + Z @ U, lower=True, check_finite=False)
        Minv = cho_solve((Lm, True), np.eye(n), check_finite=False)
        self.Ainv -= U @ Minv @ U.T
        self.Ainv = 0.5 * (self.Ainv + self.Ainv.T)
        self.A += Z.T @ Z
        self.logdet += 2.0 * float(np.log(np.diag(Lm)).sum())
        self.nobs += n
        self.since_refresh += n
        if y is None:
            for k, z in zip(keys, Z):
                self.pending[k] = z
        else:
            y = np.asarray(y, dtype=np.float64)
            self.b0 += Z.T @ y
            self.s1 += Z.sum(axis=0)
            self.ysum += float(y.sum())
            self.nlab += n
        return U, Minv

    def width_drop(self, Q, U, Minv, batch=65536):
        """Decrease of width² at the feature rows Q caused by the batch that returned (U, M⁻¹)."""
        out = np.empty(len(Q))
        U32 = U.astype(np.float32)
        for s in range(0, len(Q), batch):
            P = (Q[s:s + batch] @ U32).astype(np.float64)
            out[s:s + batch] = self.lam * np.einsum("ij,ij->i", P @ Minv, P)
        return out

    def set_label(self, key, y):
        z = self.pending.pop(key)
        self.b0 += z * float(y)
        self.s1 += z
        self.ysum += float(y)
        self.nlab += 1

    def refresh(self):
        L = cholesky(self.A, lower=True, check_finite=False)
        self.Ainv = cho_solve((L, True), np.eye(self.r), check_finite=False)
        self.since_refresh = 0

    def width2(self, Z, batch=8192):
        out = np.empty(len(Z))
        for s in range(0, len(Z), batch):
            z = np.asarray(Z[s:s + batch], dtype=np.float64)
            out[s:s + batch] = (np.maximum(1.0 - np.einsum("ij,ij->i", z, z), 0.0)
                                + self.lam * np.einsum("ij,ij->i", z @ self.Ainv, z))
        return out

    def snapshot(self):
        return {"Ainv": self.Ainv.copy(), "theta": self.theta(), "ybar": self.ybar(), "lam": self.lam}


def snap_predict(S, Z):
    """Mean and width of a frozen snapshot of an OnlineKRR."""
    Z = np.asarray(Z, dtype=np.float64)
    mean = S["ybar"] + Z @ S["theta"]
    var = np.maximum(1.0 - np.einsum("ij,ij->i", Z, Z), 0.0) + S["lam"] * np.einsum("ij,ij->i", Z @ S["Ainv"], Z)
    return mean, np.sqrt(np.maximum(var, 0.0))


def phi(d):
    """The single-turn mismatch φ(d) = d, as float32 on [0, 1]."""
    return np.clip(np.asarray(d, dtype=np.float32), 0.0, 1.0)


def sparse_reverse_greedy(w, c, cols, vals, n_cols, k):
    """Reverse Greedy for Σ_i w_i min{c_i, min_{j ∈ M} L_ij} when L_ij = vals[i, a] for j = cols[i, a] and
    L_ij = 1 elsewhere (c_i ≤ 1). Same removals and tie-breaking as the dense version, with a lazy heap."""
    w = np.asarray(w, np.float64)
    c = np.asarray(c, np.float64)
    cols = np.asarray(cols, np.int64)
    vals = np.asarray(vals, np.float32).astype(np.float64)
    n = len(w)
    ok = (w[:, None] > 0) & (cols >= 0) & (vals < c[:, None])
    e_row, a = np.nonzero(ok)
    e_col, e_val = cols[e_row, a], vals[e_row, a]
    o = np.lexsort((e_val, e_row * np.int64(n_cols) + e_col))          # dedupe (row, col): keep the lowest value
    first = np.ones(len(o), bool)
    first[1:] = np.diff((e_row * np.int64(n_cols) + e_col)[o]) != 0
    e_row, e_col, e_val = e_row[o][first], e_col[o][first], e_val[o][first]
    o = np.lexsort((e_col, e_val, e_row))                                # per row: by value, then column
    e_row, e_col, e_val = e_row[o], e_col[o], e_val[o]
    E = len(e_row)
    ptr = np.r_[0, np.cumsum(np.bincount(e_row, minlength=n))].astype(np.int64)
    end_a = ptr[1:]
    pos_b_a = ptr[:-1].copy()
    pos_s_a = pos_b_a + 1
    oc = np.argsort(e_col, kind="stable")
    cr_ptr = np.r_[0, np.cumsum(np.bincount(e_col, minlength=n_cols))].astype(np.int64).tolist()
    cr_rows = e_row[oc].tolist()
    ev_a = np.r_[e_val, 0.0]
    ecol_a = np.r_[e_col, -1]
    hb = pos_b_a < end_a
    hs = pos_s_a < end_a
    bv = ev_a[np.minimum(pos_b_a, E)]
    sv = np.where(hs, ev_a[np.minimum(pos_s_a, E)], c)
    contrib_a = np.where(hb, w * (np.minimum(c, sv) - np.minimum(c, bv)), 0.0)
    bcol_a = np.where(hb, ecol_a[np.minimum(pos_b_a, E)], -1)
    scol_a = np.where(hs, ecol_a[np.minimum(pos_s_a, E)], -1)
    inc_a = np.bincount(bcol_a[hb], weights=contrib_a[hb], minlength=n_cols).astype(np.float64)
    npos_a = np.bincount(bcol_a[hb], weights=(contrib_a[hb] > 0).astype(float), minlength=n_cols)
    ev, ecol = ev_a.tolist(), ecol_a.tolist()
    wl, cl = w.tolist(), c.tolist()
    end, pos_b, pos_s = end_a.tolist(), pos_b_a.tolist(), pos_s_a.tolist()
    bcol, scol, contrib = bcol_a.tolist(), scol_a.tolist(), contrib_a.tolist()
    inc, npos = inc_a.tolist(), npos_a.tolist()
    active = [True] * int(n_cols)
    heap = [(v, j) for j, v in enumerate(inc)]
    heapq.heapify(heap)
    n_active, target = int(n_cols), min(int(k), int(n_cols))
    while n_active > target:
        while True:
            v, r = heapq.heappop(heap)
            if active[r] and v == inc[r]:
                break
        active[r] = False
        inc[r] = float("inf")
        n_active -= 1
        aff = [i for i in cr_rows[cr_ptr[r]:cr_ptr[r + 1]] if bcol[i] == r or scol[i] == r]
        if not aff:
            continue
        touched = []
        for i in aff:
            j = bcol[i]
            if j >= 0:
                inc[j] -= contrib[i]
                npos[j] -= float(contrib[i] > 0)
                touched.append(j)
        for i in aff:
            e = end[i]
            p = pos_b[i]
            while p < e and not active[ecol[p]]:
                p += 1
            pos_b[i] = p
            q = pos_s[i] if pos_s[i] > p + 1 else p + 1
            while q < e and not active[ecol[q]]:
                q += 1
            pos_s[i] = q
            if p < e:
                ci = cl[i]
                bvi = ev[p]
                svi = ev[q] if q < e else ci
                contrib[i] = wl[i] * ((ci if ci < svi else svi) - (ci if ci < bvi else bvi))
                bcol[i] = ecol[p]
            else:
                contrib[i] = 0.0
                bcol[i] = -1
            scol[i] = ecol[q] if q < e else -1
        for i in aff:
            j = bcol[i]
            if j >= 0:
                inc[j] += contrib[i]
                npos[j] += float(contrib[i] > 0)
                touched.append(j)
        for j in set(touched):
            if active[j]:
                if npos[j] <= 0:
                    inc[j] = 0.0                              # exact zeros (no float drift in the tie-breaking)
                heapq.heappush(heap, (inc[j], j))
    return np.flatnonzero(np.asarray(active, bool))


# ---------------------------------------------------------------------------------------------
# CLCB-LS-MT
# ---------------------------------------------------------------------------------------------

class Net:
    """What the learners on one state space share: the growing ε-net, the cell of every past request,
    the graph between centers (A_ij = 1 iff j is among the 16 nearest centers of i) and the feature rows
    of the kernel models. It depends only on the states, never on serving decisions."""

    def __init__(self, name, X, stream, eps, cost_map, risk_map=None, pair_features=None):
        self.name, self.eps, self.K = name, float(eps), K_GRAPH
        self.stream = stream
        self.XS = np.ascontiguousarray(np.asarray(X[stream.states], dtype=np.float32))   # stream order
        self.thr = 1.0 - 2.0 * self.eps ** 2          # d <= ε  <=>  cosine >= thr
        self.cost_map, self.risk_map, self.pair_features = cost_map, risk_map, pair_features
        self.m, self.cap, self.n_seen = 0, 0, 0
        self.cell = np.full(stream.T, -1, dtype=np.int64)
        self.cell_sim = np.full(stream.T, -np.inf, dtype=np.float32)
        self._grow(1024)

    def _grow(self, need):
        if need <= self.cap:
            return
        cap = max(int(need), 2 * self.cap, 1024)

        def ext(name, shape, fill, dtype):
            b = np.full(shape, fill, dtype=dtype)
            if self.cap:
                b[: self.cap] = getattr(self, name)[: self.cap]
            setattr(self, name, b)
        ext("C", (cap, self.XS.shape[1]), 0.0, np.float32)
        ext("center_state", (cap,), -1, np.int64)
        ext("nbr", (cap, self.K), -1, np.int64)
        ext("nbr_sim", (cap, self.K), -np.inf, np.float32)
        ext("slot_ver", (cap, self.K), 0, np.int64)
        ext("Zc", (cap, self.cost_map.r), 0.0, np.float32)
        if self.risk_map is not None:
            ext("Zr", (cap, self.K, self.risk_map.r), 0.0, np.float32)
        self.cap = cap

    def nearest(self, pos):
        """Cell, cosine and coverage (d <= ε) of the requests at stream positions pos."""
        n = len(pos)
        if self.m == 0:
            return np.full(n, -1), np.full(n, -np.inf, np.float32), np.zeros(n, bool)
        S = self.XS[pos] @ self.C[: self.m].T
        j = np.argmax(S, axis=1)
        s = S[np.arange(n), j]
        return j, s, s >= self.thr

    def weights(self):
        return np.bincount(self.cell[: self.n_seen], minlength=self.m)[: self.m] / max(self.n_seen, 1)

    def end_conversation(self, pos):
        """Greedy ε-net update with the conversation's states; cells of every seen request; graph; features."""
        Xs = self.XS[pos]
        m0 = self.m
        mx = (Xs @ self.C[:m0].T).max(axis=1) if m0 else np.full(len(pos), -2.0, np.float32)
        new_local = []
        for i in range(len(pos)):
            if mx[i] >= self.thr:
                continue
            if new_local and float(np.max(Xs[new_local] @ Xs[i])) >= self.thr:
                continue
            new_local.append(i)
        n_new = len(new_local)
        if n_new:
            self._grow(m0 + n_new)
            ids = np.arange(m0, m0 + n_new)
            self.C[ids] = Xs[new_local]
            self.center_state[ids] = self.stream.states[pos[new_local]]
            self.Zc[ids] = self.cost_map(Xs[new_local])
            self.m = m0 + n_new
        m = self.m
        S = Xs @ self.C[:m].T
        j = np.argmax(S, axis=1)
        self.cell[pos] = j
        self.cell_sim[pos] = S[np.arange(len(pos)), j]
        if n_new and self.n_seen:
            Sp = self.XS[: self.n_seen] @ self.C[m0:m].T
            best, arg = Sp.max(axis=1), Sp.argmax(axis=1)
            upd = np.flatnonzero(best > self.cell_sim[: self.n_seen])
            self.cell[upd] = m0 + arg[upd]
            self.cell_sim[upd] = best[upd]
        self.n_seen = int(pos[-1]) + 1
        if n_new:
            self._edge_features(self._update_graph(m0))
        return n_new

    def _update_graph(self, m0):
        m, K = self.m, self.K
        new = np.arange(m0, m)
        changed = []
        S_new = self.C[new] @ self.C[:m].T
        k = min(K, m)
        for r_, i in enumerate(new):
            s = S_new[r_]
            top = np.argpartition(-s, k - 1)[:k] if k < m else np.arange(m)
            top = top[np.lexsort((top, -s[top]))]
            self.nbr[i, :] = -1
            self.nbr_sim[i, :] = -np.inf
            self.nbr[i, :k] = top
            self.nbr_sim[i, :k] = s[top]
            self.slot_ver[i, :k] += 1
            changed += [(i, a) for a in range(k)]
        if m0:
            S_old = self.C[:m0] @ self.C[new].T
            rows = np.flatnonzero((S_old > self.nbr_sim[:m0, K - 1][:, None]).any(axis=1))
            for i in rows:
                have = self.nbr[i] >= 0
                ids = np.r_[self.nbr[i][have], new]
                sims = np.r_[self.nbr_sim[i][have], S_old[i]]
                o = np.lexsort((ids, -sims))[:K]
                old = self.nbr[i].copy()
                self.nbr[i, :] = -1
                self.nbr_sim[i, :] = -np.inf
                self.nbr[i, :len(o)] = ids[o]
                self.nbr_sim[i, :len(o)] = sims[o]
                for a in np.flatnonzero(self.nbr[i] != old):
                    self.slot_ver[i, a] += 1
                    if self.nbr[i, a] >= 0:
                        changed.append((i, a))
        return np.asarray(changed, dtype=np.int64).reshape(-1, 2)

    def _edge_features(self, slots):
        if self.risk_map is None or not len(slots):
            return
        ii, aa = slots[:, 0], slots[:, 1]
        self.Zr[ii, aa] = self.risk_map(self.pair_features(self.center_state[ii], self.center_state[self.nbr[ii, aa]]))


class CLCB:
    """CLCB-LS-MT (risk='learned', session=True), or the single-turn learner on the request net
    (risk='phi': the known mismatch γ̂ φ(d); cost learned on the full cost). Each conversation is served
    with the snapshot of the last switch; data grow at every request; switches happen at conversation
    ends when a center was added, an arrival stage ended, or the cost or risk information grew by ln 2."""

    def __init__(self, family, w, net, budget, cost, risk, gamma=PRICE, single_turn=False, session=False):
        self.family, self.w, self.net, self.budget = family, w, net, str(budget)
        self.k = None if self.budget == ALL else int(self.budget)
        T = net.stream.T
        self.single_turn, self.session = single_turn, session
        self.cost_target = w.cost if single_turn else w.cost_out
        self.cin = np.zeros(w.n) if single_turn else w.cost_in
        self.beta_c, self.beta_r, self.gamma = cost.beta, risk.beta, float(gamma)
        self.pair_features = net.pair_features
        self.cost = OnlineKRR(net.cost_map.r, cost.ridge, COST_PRIOR, PRIOR_COUNT)
        self.risk = None if single_turn else OnlineKRR(net.risk_map.r, risk.ridge, RISK_PRIOR, PRIOR_COUNT)
        self.var_c = np.zeros(0)
        self.var_r = np.zeros((0, net.K))
        self.ver_r = np.zeros((0, net.K), dtype=np.int64)
        self.snap = None
        self.reuse = np.zeros(T, dtype=bool)
        self.cand = np.full(T, -1, dtype=np.int64)
        self.session_hit = np.zeros(T, dtype=bool)
        self.queue_cost, self.queue_risk, self.pending = [], [], []
        self.stage_start = 0
        self.ld_c0 = self.ld_r0 = 0.0
        self.switches, self.labels_missing, self._trig = Counter(), 0, []
        self.entries = 0

    def _sync(self):
        net, m = self.net, self.net.m
        if len(self.var_c) < m:
            new = np.arange(len(self.var_c), m)
            self.var_c = np.r_[self.var_c, self.cost.width2(net.Zc[new])]
        if self.risk is not None:
            if len(self.var_r) < m:
                add = m - len(self.var_r)
                self.var_r = np.vstack([self.var_r, np.ones((add, net.K))])
                self.ver_r = np.vstack([self.ver_r, np.full((add, net.K), -1, dtype=np.int64)])
            ii, aa = np.nonzero((self.ver_r[:m] != net.slot_ver[:m]) & (net.nbr[:m] >= 0))
            if len(ii):
                self.var_r[ii, aa] = self.risk.width2(net.Zr[ii, aa])
                self.ver_r[ii, aa] = net.slot_ver[ii, aa]

    def _risk_low(self, t, c):
        mu, sd = snap_predict(self.snap["risk"], self.net.risk_map(self.pair_features(t, c)))
        return np.minimum(1.0, self.gamma * np.maximum(mu - self.beta_r * sd - 0.0, 0.0))

    def _cost_low(self, pos):
        states = self.net.stream.states[pos]
        mc, sc = snap_predict(self.snap["cost"], self.net.cost_map(self.net.XS[pos]))
        return np.clip(self.cin[states] + mc - self.beta_c * sc, 0.0, 1.0)

    def decide(self, pos, covered):
        """The cached candidate with the lowest R̲ for every covered request (with the current snapshot):
        (idx, cand, ρ) for the requests pos[idx], or None."""
        S = self.snap
        idx = np.flatnonzero(covered)
        if S is None or not len(idx):
            return None
        n = len(idx)
        Sim = self.net.XS[pos[idx]] @ S["cache_X"].T
        kk = min(K_SERVE, Sim.shape[1])
        top = (np.argpartition(-Sim, kk - 1, axis=1)[:, :kk] if kk < Sim.shape[1]
               else np.tile(np.arange(Sim.shape[1]), (n, 1)))
        cs = S["cache_state"][top]
        if self.single_turn:
            dd = np.sqrt(np.maximum(2.0 - 2.0 * np.clip(np.take_along_axis(Sim, top, axis=1), -1, 1), 0.0)) / 2.0
            R = np.minimum(1.0, self.gamma * phi(dd).astype(np.float64))
        else:
            states = self.net.stream.states[pos[idx]]
            R = self._risk_low(np.repeat(states, kk), cs.ravel()).reshape(n, kk)
        a = np.argmin(R, axis=1)
        return idx, cs[np.arange(n), a], R[np.arange(n), a]

    def serve(self, pos, covered):
        """Serve one conversation: reuse iff c̲(z) > ρ*, the lowest R̲ over the cached candidates and (with
        a session buffer) the conversation's earlier requests that called the LLM."""
        states = self.net.stream.states[pos]
        n = len(pos)
        if self.snap is not None:
            out = self.decide(pos, covered)
            rho, cand = np.full(n, np.inf), np.full(n, -1, dtype=np.int64)
            reuse = np.zeros(n, dtype=bool)
            if out is not None:
                idx, c0, r0 = out
                rho[idx], cand[idx] = r0, c0
                if not self.session:
                    reuse[idx] = c0 >= 0
                    reuse[idx] &= self._cost_low(pos[idx]) > r0
            if self.session:
                c_low = self._cost_low(pos)
                prev, sess = [], np.zeros(n, dtype=bool)
                X = self.net.XS
                for i in range(n):
                    if prev:
                        pv = np.asarray(prev, dtype=np.int64)
                        d = np.sqrt(np.maximum(2.0 - 2.0 * np.clip(X[pv] @ X[pos[i]], -1.0, 1.0), 0.0)) / 2.0
                        ok = pv[d <= SESSION_MAX_DIST]
                        if len(ok):
                            cs = self.net.stream.states[ok]
                            R = self._risk_low(np.full(len(cs), states[i]), cs)
                            a = int(np.argmin(R))
                            if R[a] < rho[i]:
                                rho[i], cand[i], sess[i] = float(R[a]), int(cs[a]), True
                    if not (cand[i] >= 0 and c_low[i] > rho[i]):
                        prev.append(int(pos[i]))       # an LLM call: its response joins the session buffer
                reuse = (cand >= 0) & (c_low > rho)
                self.session_hit[pos[reuse & sess]] = True
            self.reuse[pos] = reuse
            self.cand[pos[reuse]] = cand[reuse]
            if self.risk is not None:                  # every reuse reveals its label (q = 1)
                self.queue_risk += list(zip(pos[reuse].tolist(), states[reuse].tolist(), cand[reuse].tolist()))
        self.queue_cost += pos[~self.reuse[pos]].tolist()

    def end_conversation(self, pos, n_new):
        net = self.net
        self._sync()
        m = net.m
        states = net.stream.states
        if self.queue_cost:
            q = np.asarray(self.queue_cost)
            U, Minv = self.cost.add_batch(net.cost_map(net.XS[q]), y=self.cost_target[states[q]])
            if m:
                self.var_c[:m] -= self.cost.width_drop(net.Zc[:m], U, Minv)
            self.queue_cost = []
        if self.risk is not None and self.queue_risk:
            tt = np.array([t for _, t, _ in self.queue_risk])
            cc = np.array([c for _, _, c in self.queue_risk])
            U, Minv = self.risk.add_batch(net.risk_map(self.pair_features(tt, cc)),
                                          keys=[p for p, _, _ in self.queue_risk])
            if m:
                self.var_r[:m] -= self.risk.width_drop(net.Zr[:m].reshape(-1, net.risk_map.r), U, Minv).reshape(m, net.K)
            self.pending += self.queue_risk
            self.queue_risk = []
        if self.cost.since_refresh >= REFRESH_EVERY:
            self.cost.refresh()
            self.var_c[:m] = self.cost.width2(net.Zc[:m])
        if self.risk is not None and self.risk.since_refresh >= REFRESH_EVERY:
            self.risk.refresh()
            ii, aa = np.nonzero(net.nbr[:m] >= 0)
            self.var_r[ii, aa] = self.risk.width2(net.Zr[ii, aa])
        seen = int(pos[-1]) + 1
        trig = []
        if n_new:
            trig.append("new center")
        if seen - self.stage_start >= 1 + np.sqrt(net.stream.T * self.stage_start):
            trig.append("arrival stage")
            self.stage_start = seen
        if self.cost.logdet - self.ld_c0 >= np.log(2.0):
            trig.append("cost information")
        if self.risk is not None and self.risk.logdet - self.ld_r0 >= np.log(2.0):
            trig.append("risk information")
        self._trig = trig
        return bool(trig) and m > 0

    def resolve_labels(self):
        """Attach the judge's verdicts to the pending risk inputs (1 = unacceptable; missing → 1)."""
        if not self.pending:
            return
        t = np.array([t for _, t, _ in self.pending])
        c = np.array([c for _, _, c in self.pending])
        acc = self.w.labels.lookup(t, c)
        self.labels_missing += int(np.isnan(acc).sum())
        for (p, _, _), a in zip(self.pending, acc):
            self.risk.set_label(p, 1.0 if np.isnan(a) else 1.0 - a)
        self.pending = []

    def switch(self):
        """Recompute the snapshot: c̲ of the centers, R̲ on the graph, and the cache by Reverse Greedy."""
        net = self.net
        m, K = net.m, net.K
        sd_c = np.sqrt(np.maximum(self.var_c[:m], 0.0))
        mean_c = self.cost.ybar() + (net.Zc[:m] @ self.cost.theta().astype(np.float32)).astype(np.float64)
        c_low = np.clip(self.cin[net.center_state[:m]] + mean_c - self.beta_c * sd_c, 0.0, 1.0)
        nbr = net.nbr[:m].copy()
        if self.risk is not None:
            mu = self.risk.ybar() + (net.Zr[:m].reshape(-1, net.risk_map.r)
                                     @ self.risk.theta().astype(np.float32)).astype(np.float64).reshape(m, K)
            sd_r = np.sqrt(np.maximum(self.var_r[:m], 0.0))
            R_low = np.minimum(1.0, self.gamma * np.maximum(mu - self.beta_r * sd_r - 0.0, 0.0))
        else:
            d = np.sqrt(np.maximum(2.0 - 2.0 * np.clip(net.nbr_sim[:m], -1, 1), 0.0)) / 2.0
            R_low = np.minimum(1.0, self.gamma * phi(d))
        R_low = np.where(nbr >= 0, R_low, 1.0).astype(np.float32)
        if self.k is None or m <= self.k:
            cache = np.arange(m)
        else:
            cache = sparse_reverse_greedy(net.weights(), c_low, nbr, R_low, m, self.k)
        self.entries = len(cache)
        self.snap = {"cache_X": net.C[cache].copy(), "cache_state": net.center_state[cache].copy(),
                     "cost": self.cost.snapshot(), "risk": self.risk.snapshot() if self.risk is not None else None}
        self.ld_c0 = self.cost.logdet
        self.ld_r0 = self.risk.logdet if self.risk is not None else 0.0
        for tr in self._trig:
            self.switches[tr] += 1
        self.switches["total"] += 1


def run_learners(stream, nets, learners, log=print):
    """Replay the stream. At each conversation's end: nets update, learners add their observations, and
    the switching learners attach their pending labels and recompute their snapshots."""
    for j in range(stream.n_conv):
        pos = np.arange(stream.bounds[j], stream.bounds[j + 1])
        covered = {name: net.nearest(pos)[2] for name, net in nets.items()}
        for L in learners:
            L.serve(pos, covered[L.net.name])
        n_new = {name: net.end_conversation(pos) for name, net in nets.items()}
        for L in learners:
            if L.end_conversation(pos, n_new[L.net.name]):
                if L.risk is not None:
                    L.resolve_labels()
                L.switch()
        if (j + 1) % 200 == 0 or j + 1 == stream.n_conv:
            log(f"    {j + 1:,}/{stream.n_conv:,} conversations | centers: "
                + ", ".join(f"{name} {net.m:,}" for name, net in nets.items()))
    for L in learners:
        if L.risk is not None:
            L.resolve_labels()


# ---------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------

def lru_baseline(w, stream, kind, budget, setting):
    """ElastiCache-MT ('elasticache'), the joint-state threshold ('jointstate') or ContextCache
    ('contextcache') on the stream: every LLM response is inserted right away and a full cache evicts
    its least recently used entry. Returns (reuse, served state) per stream position."""
    T, states = stream.T, stream.states
    cap = None if budget == ALL else int(budget)
    E = {"elasticache": w.E_context, "jointstate": w.X}.get(kind, w.E_query)
    ES = np.ascontiguousarray(np.asarray(E[states], dtype=np.float32))
    in_cache = np.zeros(T, dtype=bool)
    last_used = np.full(T, -1, dtype=np.int64)
    slots = np.full(cap if cap else T, -1, dtype=np.int64)
    n_cached = 0
    served = np.zeros(T, dtype=bool)
    cand = np.full(T, -1, dtype=np.int64)
    by_ctx = defaultdict(list)
    prefix = np.vstack([np.zeros((1, w.E_query.shape[1])), np.cumsum(w.E_query[w.sorted_order].astype(np.float64), axis=0)])
    n_upto = w.pos_in_conv + 1

    def window_mean(g, wl):
        r = w.rank_of[np.asarray(g, dtype=int)]
        return (prefix[r + 1] - prefix[r + 1 - wl]) / wl[:, None]

    for b0 in range(0, T, BASELINE_BLOCK):
        b1 = min(T, b0 + BASELINE_BLOCK)
        S = ES[b0:b1] @ ES[:b1].T
        for t in range(b0, b1):
            hit, j = False, -1
            g = int(states[t])
            if n_cached:
                row = S[t - b0, :t]
                if kind in ("elasticache", "jointstate"):
                    vals = np.where(in_cache[:t], row, -np.inf)
                    j = int(np.argmax(vals))
                    hit = vals[j] >= setting[0]
                else:
                    ex = [p for p in by_ctx.get(int(w.context_code[g]), []) if in_cache[p]]
                    if ex:
                        j, hit = ex[0], True
                    elif (idx := np.flatnonzero(in_cache[:t])).size:
                        vals = row[idx]
                        K = min(100, len(idx))
                        top = np.argpartition(-vals, K - 1)[:K] if K < len(idx) else np.arange(len(idx))
                        top = top[np.argsort(-vals[top], kind="stable")]
                        cp, cv = idx[top], vals[top]
                        _, first = np.unique(w.qa_code[states[cp]], return_index=True)
                        keep = np.sort(first)
                        cp, cv = cp[keep], cv[keep]
                        cg = states[cp]
                        wl = np.minimum(5, np.minimum(n_upto[g], n_upto[cg]))
                        A, B = window_mean(np.full(len(cg), g), wl), window_mean(cg, wl)
                        num = np.sum(A * B, axis=1)
                        den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
                        cs = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
                        b = np.lexsort((-cv, -cs))[0]
                        j = int(cp[b])
                        hit = (cv[b] >= setting[0]) and (cs[b] >= setting[1])
            if hit:
                served[t] = True
                cand[t] = states[j]
                last_used[j] = t
                continue
            if cap is not None and n_cached >= cap:                 # LRU eviction
                k_ = int(np.argmin(last_used[slots[:n_cached]]))
                victim = int(slots[k_])
                in_cache[victim] = False
                if kind == "contextcache":
                    by_ctx[int(w.context_code[states[victim]])].remove(victim)
                slots[k_] = t
            else:
                slots[n_cached] = t
                n_cached += 1
            in_cache[t] = True
            last_used[t] = t
            if kind == "contextcache":
                by_ctx[int(w.context_code[g])].append(t)
    return served, cand


class VCacheOnline:
    """vCache on the stream (cold start, LRU eviction at the budget). The nearest entry of every request
    (by the context-string embedding) is touched; the request is served if the entry's correctness bound
    allows it, otherwise it explores: the LLM is called, the judge's verdict becomes an observation of
    the entry, and a wrong cached response makes the new response an entry."""

    def __init__(self, w, stream, budget, delta, U):
        self.w, self.stream, self.delta, self.U = w, stream, float(delta), U
        self.k = None if budget == ALL else int(budget)
        cap = stream.T if self.k is None else self.k
        self.ES = np.ascontiguousarray(np.asarray(w.E_context[stream.states], dtype=np.float32))
        self.pos = np.full(cap, -1, dtype=np.int64)
        self.emb = np.zeros((cap, self.ES.shape[1]), dtype=np.float32)
        self.last = np.zeros(cap, dtype=np.int64)
        self.obs = [None] * cap
        self.oh = [0] * cap
        self.n = self.clock = 0
        self.served = np.zeros(stream.T, bool)
        self.cand = np.full(stream.T, -1, dtype=np.int64)
        self.fits, self.stats = {}, Counter()

    def _insert(self, p):
        self.clock += 1
        if self.k is not None and self.n >= self.k:
            j = int(np.argmin(self.last[: self.n]))
            self.stats["evicted"] += 1
        else:
            j = self.n
            self.n += 1
        self.pos[j], self.emb[j], self.last[j] = p, self.ES[p], self.clock
        self.obs[j], self.oh[j] = [(0.0, 0), (1.0, 1)], hash((int(p), -1))
        self._changed.add(j)

    def run(self):
        states = self.stream.states
        for r0 in range(0, self.stream.T, VC_ROUND):
            P = np.arange(r0, min(self.stream.T, r0 + VC_ROUND))
            S0, n0, self._changed = (self.ES[P] @ self.emb[: self.n].T if self.n else None), self.n, set()
            for i, p in enumerate(P):
                self.clock += 1
                n, j = self.n, -1
                if n:
                    s = np.full(n, -np.inf, dtype=np.float32)
                    if S0 is not None:
                        s[:n0] = S0[i]
                    for c in self._changed:
                        s[c] = self.emb[c] @ self.ES[p]
                    j = int(np.argmax(s))
                    sv = float(s[j])
                    if not np.isfinite(sv):
                        j = -1
                if j < 0:
                    self._insert(p)
                    continue
                sv = round(sv, 3)
                self.last[j] = self.clock
                o, explore = self.obs[j], True
                if len(o) >= 6:
                    key = (int(self.pos[j]), self.oh[j])
                    if key not in self.fits:
                        self.fits[key] = vc_fit(o)
                    prm = self.fits[key]
                    if prm is not None:
                        explore = bool(self.U[p] <= vc_tau(vc_alpha(prm, sv), self.delta))
                if not explore:
                    self.served[p], self.cand[p] = True, int(states[self.pos[j]])
                    continue
                self.stats["explorations"] += 1
                if sv < VC_FLOOR:
                    y = 0
                else:
                    a = self.w.labels.lookup([states[p]], [states[self.pos[j]]])[0]
                    if np.isnan(a):                       # no verdict: vCache skips the update
                        self.stats["label missing"] += 1
                        continue
                    y = int(a)
                o.append((sv, y))
                self.oh[j] = hash((self.oh[j], sv, y))
                if y == 0:
                    self._insert(p)
        return self.served, self.cand


# ---------------------------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------------------------

def offline_settings(path):
    """The offline selections (results/<workload>/offline_loss.csv): {(method, budget): params}."""
    t = pd.read_csv(path, dtype={"budget": str})
    return {(r.method, str(r.budget)): json.loads(r.params) for r in t.itertuples()}


class Evaluation:
    """Per-request loss in stream order and conversation-bootstrap intervals over the whole stream."""

    def __init__(self, w, stream):
        self.w, self.stream = w, stream
        self.conv_cost = np.bincount(stream.conv, weights=stream.cost, minlength=stream.n_conv)
        n = stream.n_conv
        self.W = np.random.default_rng(SEED + 41).multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)

    def accepted(self, reuse, cand):
        acc = np.full(self.stream.T, np.nan)
        acc[reuse] = self.w.labels.lookup(self.stream.states[reuse], cand[reuse])
        return acc

    def loss(self, reuse, acc):
        """c(z) if the LLM is called, min{1, γ} if an unacceptable or unjudged response is served."""
        return np.where(reuse, np.where(acc == 1.0, 0.0, min(1.0, PRICE)), self.stream.cost)

    def ci(self, v):
        s = np.bincount(self.stream.conv, weights=v, minlength=self.stream.n_conv)
        boot = 100 * (self.W @ s) / (self.W @ self.conv_cost)
        return 100 * s.sum() / self.conv_cost.sum(), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def run(w, cost, risk, out, log=print):
    stream = Stream(w)
    sel = offline_settings(out / "offline_loss.csv")
    budgets = list(ONLINE_BUDGETS)
    eps_by_budget = [sel[("CUCB-SC-MT", str(k))][0] for k in BUDGETS if ("CUCB-SC-MT", str(k)) in sel]
    counts = pd.Series(eps_by_budget).value_counts()
    eps = float(min(counts[counts == counts.max()].index))         # the most common offline ε (ties: smaller)
    log(f"  stream: {stream.n_conv:,} conversations, {stream.T:,} requests | CLCB-LS-MT ε = {eps:g}")

    pick = np.random.default_rng(SEED).choice(stream.T, size=min(stream.T, LANDMARK_SAMPLE * 1024), replace=False)
    pair_features = lambda t, c: risk.standardize(risk.features.raw(t, c))
    nets = {"state": Net("state", w.X, stream, eps,
                         NystromMap(w.X[stream.states[pick]], 1024, cost.scale * cost.gamma0),
                         NystromMap(risk.F_train, 512, risk.gamma), pair_features)}
    query_map = NystromMap(w.E_query[stream.states[pick]], 1024, cost.scale / median_sq_distance(w.E_query[w.cal]))
    learners = []
    for b in budgets:
        learners.append(CLCB("CLCB-LS-MT", w, nets["state"], b, cost, risk, session=True))
    for b in budgets:
        e, g = sel[("Single-turn", b)]
        name = f"query ε={e:g}"
        if name not in nets:
            nets[name] = Net(name, w.E_query, stream, e, query_map)
        learners.append(CLCB("Single-turn", w, nets[name], b, cost, risk, gamma=g, single_turn=True))
    log(f"  CLCB-LS-MT and single-turn learners at budgets {budgets}")
    run_learners(stream, nets, learners, log=log)

    runs = {}
    for L in learners:
        runs[(L.family, L.budget)] = (L.reuse, L.cand, L.entries)
    for kind, method in [("elasticache", "ElastiCache-MT"), ("contextcache", "ContextCache"),
                         ("jointstate", "Joint-state threshold")]:
        for b in budgets:
            reuse, cand = lru_baseline(w, stream, kind, b, sel[(method, b)])
            runs[(method, b)] = (reuse, cand, None)
        log(f"  {method}: done")
    U = np.random.default_rng(SEED + 151).random(stream.T)
    for b in budgets:
        reuse, cand = VCacheOnline(w, stream, b, sel[("vCache", b)][0], U).run()
        runs[("vCache", b)] = (reuse, cand, None)
    log("  vCache: done")

    ev = Evaluation(w, stream)
    rows, losses = [], {}
    nets_m = nets["state"].m
    for method in METHODS:
        for b in budgets:
            reuse, cand, entries = runs[(method, b)]
            acc = ev.accepted(reuse, cand)
            loss = ev.loss(reuse, acc)
            losses[(method, b)] = loss
            point, lo, hi = ev.ci(loss)
            n_acc, n_rej = int((acc == 1.0).sum()), int((acc == 0.0).sum())
            c = stream.cost
            rows.append({
                "method": method, "budget": b, "loss": point, "loss_lo": lo, "loss_hi": hi,
                "hits": int(reuse.sum()), "unjudged": int((reuse & np.isnan(acc)).sum()),
                "precision": 100 * n_acc / max(n_acc + n_rej, 1), "false_reuse": 100 * n_rej / stream.T,
                "safe_cost_avoided": 100 * c[acc == 1.0].sum() / c.sum(),
                "hit_cost_ratio": c[reuse].mean() / c.mean() if reuse.any() else np.nan,
                "entries": entries if method != "CLCB-LS-MT" or b != ALL else nets_m})
            log(f"  {method:22s} {b:>4s}  loss {point:5.1f} [{lo:.1f}, {hi:.1f}]  hits {int(reuse.sum()):,}"
                + (f" ({rows[-1]['unjudged']} unjudged)" if rows[-1]["unjudged"] else ""))
    table = pd.DataFrame(rows)
    table.to_csv(out / "online_loss.csv", index=False)
    np.savez_compressed(out / "online_losses.npz", cost=stream.cost,
                        **{f"{m}|{b}": v for (m, b), v in losses.items()})
    paired = []
    for b in budgets:
        for m in METHODS[1:]:
            d, lo, hi = ev.ci(losses[("CLCB-LS-MT", b)] - losses[(m, b)])
            paired.append({"budget": b, "versus": m, "versus_budget": b, "delta": d, "lo": lo, "hi": hi})
    d, lo, hi = ev.ci(losses[("CLCB-LS-MT", ALL)] - losses[("CLCB-LS-MT", "1000")])
    paired.append({"budget": ALL, "versus": "CLCB-LS-MT", "versus_budget": "1000", "delta": d, "lo": lo, "hi": hi})
    pd.DataFrame(paired).to_csv(out / "online_paired_differences.csv", index=False)
    missing = sum(L.labels_missing for L in learners)
    log(f"  wrote {out / 'online_loss.csv'}" + (f" ({missing} risk labels missing, counted unacceptable)" if missing else ""))
