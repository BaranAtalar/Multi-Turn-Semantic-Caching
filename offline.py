"""Offline experiments (Figure 3a, 3c): every method learns its cache from logged conversations.

Each workload is split by conversation into calibration (1,800), validation (200) and test (200)
conversations. Every method's knobs are chosen to minimize the validation loss with a cache built
from calibration conversations; the chosen setting is then rebuilt from calibration and validation
conversations together (the source pool) and evaluated on the test conversations.

Loss of a request z at price γ = 1: c(z) if the LLM is called, 1 if a cached response is reused and
the judge rejects it (or the pair has no judge label), and 0 for an acceptable reuse. Results are
reported as a percentage of the no-cache loss Σ c(z).
"""
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from caching import assign, cell_weights, distance, epsilon_net, lfu_cache, lfu_order, reverse_greedy, top1
from data import SEED
from models import NystromKRR, median_sq_distance

PRICE = 1.0                    # loss of an unacceptable reuse
BUDGETS = [100, 200, 300, 500, 1000]
ALL = "all"                    # no budget: every source state (baselines) or every ε-net center
MAX_UNJUDGED = 0.01            # settings with more unjudged validation hits than this fraction are skipped
N_BOOT = 2000                  # cluster-bootstrap resamples (over test conversations)

# CUCB-SC-MT
GAMMAS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0]   # risk multiplier γ̂ in R̄ = min{1, γ̂ (r̂ + β_r σ_r)}
EPS_SEARCH = np.round(np.arange(0.10, 0.501, 0.01), 2)          # ε* = smallest ε covering 90% of validation states
EPS_COVERAGE, EPS_MIN_SIZE, EPS_MAX_SIZE = 0.90, 200, 1500       # with a net of 200 to 1,500 representatives
EPS_STEPS = 5                                                   # then ε ∈ {ε*, ε* − 0.02, ..., ε* − 0.08}
MAX_NET, MAX_NET_ALL = 5000, 8000     # larger nets are skipped (with a budget, Reverse Greedy needs an m × m matrix)
D1_FRACTION = 0.5                                               # conversations used for the net (the rest weigh it)

# Baselines' validation grids
_FINE = np.round(np.arange(0.991, 0.99901, 0.001), 3)
EC_GRID = np.unique(np.r_[np.round(np.arange(0.50, 0.991, 0.01), 2), _FINE])
CC_GRID = np.unique(np.r_[np.round(np.arange(0.50, 0.991, 0.02), 2), 0.99, _FINE])
JS_GRID = np.unique(np.round(np.r_[np.arange(0.80, 0.9901, 0.01), _FINE], 3))
EC_FLOOR, CC_FLOOR, JS_FLOOR = 0.90, 0.80, 0.80                 # budgeted grids start here
CC_TOP_K, CC_WINDOW = 100, 5
VC_DELTAS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
VC_FLOOR, VC_BATCH = 0.90, 32
ST_EPS = [0.15, 0.20, 0.25, 0.30]
ST_GAMMAS = [1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0]
ST_RANK, ST_RIDGE, ST_BETA_SCALE, ST_DELTA = 256, 1e-2, 0.10, 0.05


# ---------------------------------------------------------------------------------------------
# Decisions, loss and selection
# ---------------------------------------------------------------------------------------------

class Decisions:
    """Which requests a method serves from the cache, and with which cached state."""

    def __init__(self, w, idx, cand, reuse, entries=None):
        self.idx = np.asarray(idx, dtype=int)
        self.cand = np.asarray(cand, dtype=int)
        self.reuse = np.asarray(reuse, dtype=bool) & (self.cand >= 0)
        self.entries = entries
        self.accepted = np.full(len(self.idx), np.nan)
        self.accepted[self.reuse] = w.labels.lookup(self.idx[self.reuse], self.cand[self.reuse])
        self.cost = w.cost[self.idx]

    def summary(self):
        hit, acc = self.reuse, self.accepted
        tot = self.cost.sum()
        return {"states": len(self.idx), "hits": int(hit.sum()),
                "rejected": int((hit & (acc == 0.0)).sum()), "unjudged": int((hit & np.isnan(acc)).sum()),
                "gross_cost_avoided": self.cost[hit].sum() / tot}

    def losses(self, price=PRICE):
        """Per-request loss: the LLM cost on a miss, the price on an unacceptable or unjudged reuse."""
        unsafe = self.reuse & (self.accepted != 1.0)
        return np.where(self.reuse, 0.0, self.cost) + np.where(unsafe, min(1.0, price), 0.0)


def validation_loss(s, total_cost, price=PRICE):
    unsafe = s["rejected"] + s["unjudged"]
    return (total_cost * (1.0 - s["gross_cost_avoided"]) + min(1.0, price) * unsafe) / s["states"]


def select(sweep, total_cost):
    """The setting with the lowest validation loss (ties: fewer hits, then grid order), among the
    settings with at most 1% unjudged hits (all settings if none qualifies; unjudged hits then count
    as unacceptable). sweep: list of (setting, Decisions). Returns (setting, validation loss, fully judged)."""
    rows = []
    for setting, d in sweep:
        s = d.summary()
        rows.append((validation_loss(s, total_cost), s["hits"], len(rows), setting, s["unjudged"] <= MAX_UNJUDGED * s["hits"]))
    judged = [r for r in rows if r[4]]
    best = min(judged or rows, key=lambda r: (r[0], r[1], r[2]))
    return best[3], best[0], bool(judged)


class Bootstrap:
    """Cluster bootstrap over test conversations: a total over test requests as % of the no-cache cost."""

    def __init__(self, w):
        conv = pd.factorize(pd.Series(w.conv_ids[w.test]))[0]
        n = int(conv.max()) + 1
        self.conv, self.n = conv, n
        self.cost = np.bincount(conv, weights=w.cost[w.test], minlength=n)
        self.W = np.random.default_rng(SEED + 23).multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)

    def __call__(self, v):
        """Point estimate and 95% interval of 100 Σ v / Σ c."""
        s = np.bincount(self.conv, weights=v, minlength=self.n)
        boot = 100 * (self.W @ s) / (self.W @ self.cost)
        return 100 * s.sum() / self.cost.sum(), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


# ---------------------------------------------------------------------------------------------
# CUCB-SC-MT (Algorithm 1)
# ---------------------------------------------------------------------------------------------

def split_conversations(w, idx, seed):
    """Split the states idx by conversation into D1 (builds the ε-net) and D2 (weighs its cells)."""
    convs = pd.unique(w.conv_ids[idx[np.argsort(w.rank_of[idx])]])
    perm = np.random.default_rng(seed).permutation(len(convs))
    d1 = set(convs[perm[: int(round(D1_FRACTION * len(convs)))]])
    in_d1 = np.array([cv in d1 for cv in w.conv_ids[idx]])
    return idx[in_d1], idx[~in_d1]


def epsilon_grid(w):
    """ε* is the smallest radius whose net on calibration D1 covers 90% of the validation states
    within ε, among nets of 200 to 1,500 representatives; the grid is ε*, ε* − 0.02, ..., ε* − 0.08."""
    d1, _ = split_conversations(w, w.cal, SEED)
    coverage = {}
    for eps in EPS_SEARCH:
        reps = epsilon_net(w.X, d1, eps, max_size=EPS_MAX_SIZE)
        if reps is not None and len(reps) >= EPS_MIN_SIZE:
            coverage[float(eps)] = float(np.mean(assign(w.X, w.val, reps)[1] <= eps))
    meets = [e for e, c in coverage.items() if c >= EPS_COVERAGE]
    star = min(meets) if meets else max(coverage, key=lambda e: (coverage[e], -e))
    return sorted({round(max(star - 0.02 * i, 0.05), 2) for i in range(EPS_STEPS)}, reverse=True)


class CUCBSCMT:
    """CUCB-SC-MT for one radius ε and one phase: 'tune' builds the cache from calibration
    conversations and serves validation requests, 'final' builds it from the source pool and serves
    test requests.

    The ε-net's representatives are the candidate cache entries. Each cell is weighted by the share
    of D2 states it contains; its cost bound is c̄ of its representative; R̂_ij bounds the risk of
    serving representative i with the response of j. Reverse Greedy picks k representatives
    minimizing Σ_i w_i min{c̄_i, min_{j ∈ M} R̄_ij}. A request z is served by the cached entry with
    the lowest risk bound R̄(z, u) if that bound is below its own cost bound c̄(z).
    """

    def __init__(self, w, cost, risk, eps, phase):
        self.eps, self.phase, self.w, self.risk = eps, phase, w, risk
        pool, seed = (w.cal, SEED) if phase == "tune" else (w.src, SEED + 1)
        self.d1, self.d2 = split_conversations(w, pool, seed)
        self.reps = epsilon_net(w.X, self.d1, eps, max_size=MAX_NET_ALL)
        if self.reps is None:
            return
        self.beta_r = risk.beta
        self.cbar = cost.ucb(self.reps, phase)
        self.idx = w.val if phase == "tune" else w.test
        self.cbar_request = cost.ucb(self.idx, phase)
        self._caches, self._centers, self._requests = {}, None, None

    def usable(self, k):
        """Whether this net is used at budget k: at most MAX_NET representatives with a budget."""
        return self.reps is not None and (k == ALL or len(self.reps) <= MAX_NET)

    def cache(self, k, gamma):
        """Columns (representatives) Reverse Greedy keeps at budget k (ALL: every representative)."""
        m = len(self.reps)
        kk = m if k == ALL else int(k)
        key = (kk, float(gamma))
        if key not in self._caches:
            if kk >= m:
                self._caches[key] = np.arange(m)
            else:
                if self._centers is None:            # weights and the center risk matrix, computed once
                    weights = cell_weights(assign(self.w.X, self.d2, self.reps)[0], len(self.reps))
                    self._centers = (weights, *self.risk.matrix(self.reps, self.reps))
                weights, rhat, rsd = self._centers
                R = np.minimum(1.0, gamma * (rhat + self.beta_r * rsd + 0.0)).astype(np.float32)
                self._caches[key] = reverse_greedy(weights, self.cbar, R, kk)
        return self._caches[key]

    def decisions(self, k, gamma):
        if self._requests is None:                   # every request against every representative, computed once
            self._requests = self.risk.matrix(self.idx, self.reps)
        mu, sd = self._requests
        sel = self.cache(k, gamma)
        R = np.minimum(1.0, gamma * (mu[:, sel] + self.beta_r * sd[:, sel] + 0.0))
        j = np.argmin(R, axis=1)
        best = R[np.arange(len(j)), j]
        return Decisions(self.w, self.idx, self.reps[sel[j]], best < self.cbar_request, entries=len(sel))


# ---------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------

def context_cache_retrieve(w, cache, idx):
    """ContextCache: an identical recent context is served directly (exact tier); otherwise the top-K
    cached requests by similarity, de-duplicated by (request, answer), are ranked by the similarity of
    the mean embedding of the last 5 user requests, then by request similarity."""
    cache = np.asarray(cache, dtype=int)[np.argsort(w.rank_of[cache])]
    exact_first = {}
    for g in cache:
        exact_first.setdefault(int(w.context_code[g]), int(g))
    prefix = w.query_prefix
    Eq = w.E_query[cache]
    K = min(CC_TOP_K, len(cache))
    n = len(idx)
    cand, qsim, csim = np.full(n, -1, dtype=int), np.full(n, np.nan), np.full(n, np.nan)
    exact = np.zeros(n, dtype=bool)

    def window_mean(g, wl):
        r = w.rank_of[np.asarray(g, dtype=int)]
        return (prefix[r + 1] - prefix[r + 1 - wl]) / wl[:, None]

    for s in range(0, n, 256):
        eb = idx[s:s + 256]
        S = w.E_query[eb] @ Eq.T
        top = np.argpartition(-S, K - 1, axis=1)[:, :K]
        for i, t in enumerate(eb):
            p = s + i
            g = exact_first.get(int(w.context_code[t]))
            if g is not None:
                cand[p], qsim[p], csim[p], exact[p] = g, 1.0, 1.0, True
                continue
            cols = top[i]
            vals = S[i, cols]
            order = np.argsort(-vals, kind="stable")
            cols, vals = cols[order], vals[order]
            cg = cache[cols]
            keep = np.sort(np.unique(w.qa_code[cg], return_index=True)[1])
            cg, vals = cg[keep], vals[keep]
            wl = np.minimum(CC_WINDOW, np.minimum(w.pos_in_conv[t] + 1, w.pos_in_conv[cg] + 1))
            A, B = window_mean(np.full(len(cg), t), wl), window_mean(cg, wl)
            num = np.sum(A * B, axis=1)
            den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
            cs = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
            b = np.lexsort((-vals, -cs))[0]
            cand[p], qsim[p], csim[p] = cg[b], vals[b], cs[b]
    return cand, qsim, csim, exact


def context_cache_reuse(q, c, exact, tq, tc):
    return exact | ((np.nan_to_num(q, nan=-1) >= tq) & (np.nan_to_num(c, nan=-1) >= tc))


class SingleTurn:
    """CUCB-SC-Cont (Atalar et al., 2026) on the request alone: an ε-net of request embeddings, a cost
    UCB from KRR on the request, Reverse Greedy on γ̂ φ(d) with φ(d) = d, and reuse iff γ̂ φ(d) < UCB."""

    def __init__(self, w):
        self.w = w
        self.gamma = 1.0 / median_sq_distance(w.E_query[w.cal])
        self.models = {phase: NystromKRR(ST_RANK, ST_RIDGE, self.gamma, center=False).fit(w.E_query[p], w.cost[p])
                       for phase, p in [("tune", w.cal), ("final", w.src)]}
        self._designs, self._eval = {}, {}

    def design(self, eps, k, phase):
        key = (float(eps), k, phase)
        if key not in self._designs:
            w = self.w
            source = w.cal if phase == "tune" else w.src
            centers = epsilon_net(w.E_query, source, eps)
            weights = cell_weights(assign(w.E_query, source, centers)[0], len(centers))
            ch, cs = self.models[phase].predict(w.E_query[centers])
            beta = ST_BETA_SCALE * math.sqrt(max(1e-12, math.log(max(2 * len(centers), 2) / ST_DELTA)))
            ucb = np.clip(ch + 0.0 + beta * cs, 0.0, 1.0)
            if k == ALL or int(k) >= len(centers):
                cache = centers
            else:
                L = distance(w.E_query[centers], w.E_query[centers])
                np.fill_diagonal(L, 0.0)
                cache = centers[reverse_greedy(weights, ucb, L, int(k))]
            self._designs[key] = (cache, beta)
        return self._designs[key]

    def decisions(self, eps, k, gamma, phase):
        key = (float(eps), k, phase)
        if key not in self._eval:
            w = self.w
            cache, beta = self.design(eps, k, phase)
            idx = w.val if phase == "tune" else w.test
            S = w.E_query[idx] @ w.E_query[cache].T
            j = np.argmax(S, axis=1)
            phi = np.clip(np.sqrt(np.maximum(2.0 - 2.0 * S[np.arange(len(j)), j], 0.0)) / 2.0, 0.0, 1.0)
            ch, cs = self.models[phase].predict(w.E_query[idx])
            self._eval[key] = (idx, cache[j], phi, np.clip(ch + 0.0 + beta * cs, 0.0, 1.0), len(cache))
        idx, cand, phi, ucb, entries = self._eval[key]
        return Decisions(self.w, idx, cand, gamma * phi < ucb, entries=entries)


# vCache (Gaspar Schroeder et al., 2026), ported from the authors' code ------------------------

_VC_EPS = np.linspace(1e-6, 1 - 1e-6, 50)
_VC_Z = norm.ppf(1 - _VC_EPS)
_VC_VAR = {6: 0.035445, 7: 0.028285, 8: 0.026436, 9: 0.021349, 10: 0.019371, 11: 0.012615, 12: 0.011433,
           13: 0.010228, 14: 0.009963, 15: 0.009253, 16: 0.011674, 17: 0.013015, 18: 0.010897, 19: 0.011841,
           20: 0.013081, 21: 0.010585, 22: 0.014255, 23: 0.012058, 24: 0.013002, 25: 0.011715, 26: 0.00839,
           27: 0.008839, 28: 0.010628, 29: 0.009899, 30: 0.008033, 31: 0.00457, 32: 0.007335, 33: 0.008932,
           34: 0.00729, 35: 0.007445, 36: 0.00761, 37: 0.011423, 38: 0.011233, 39: 0.006783, 40: 0.005233,
           41: 0.00872, 42: 0.010005, 43: 0.01199, 44: 0.00977, 45: 0.01891, 46: 0.01513, 47: 0.02109,
           48: 0.01531}       # variance of the threshold estimate when the observations are perfectly separated
_VC_MIN_OBS = 6               # an entry explores until it has 6 observations (including the 2 pseudo-observations)
_VC_PSEUDO = [(0.0, 0), (1.0, 1)]


def vc_fit(obs):
    """An entry's sigmoid (threshold t̂, slope, variance of t̂) from its (similarity, correct) observations."""
    s = np.fromiter((o[0] for o in obs), float, len(obs))
    y = np.fromiter((o[1] for o in obs), int, len(obs))
    X = np.c_[np.ones(len(s)), s]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression(penalty=None, solver="lbfgs", tol=1e-8, max_iter=1000, fit_intercept=False).fit(X, y)
        intercept, gamma = (float(v) for v in lr.coef_[0])
        gamma = max(gamma, 1e-6)
        t_hat = float(np.clip(-intercept / gamma, 0.0, 1.0))
        if s[y == 1].min() > s[y == 0].max():
            var_t = _VC_VAR.get(len(s), _VC_VAR[max(_VC_VAR)])
        else:
            p = lr.predict_proba(X)[:, 1]
            H = X.T @ ((p * (1 - p))[:, None] * X)
            grad = np.array([-1.0 / gamma, intercept / gamma ** 2])
            var_t = max(0.0, float(grad @ np.linalg.inv(H) @ grad))
        return round(t_hat, 3), round(gamma, 3), var_t
    except Exception:
        return None


def vc_alpha(params, s):
    """Lower bound on the probability that the entry's response is correct at similarity s."""
    t_hat, gamma, var_t = params
    t_primes = np.clip(t_hat + _VC_Z * np.sqrt(var_t), 0.0, 1.0)
    return float(np.max((1 - _VC_EPS) * expit(gamma * (s - t_primes))))


def vc_tau(alpha, delta):
    """Exploration probability for error bound δ."""
    return np.round(1.0 - np.asarray(delta, float) / (1.0 - np.asarray(alpha, float)), 5)


class VCache:
    """A vCache cache of (at most) k entries with LRU eviction. Every request's nearest entry from
    another conversation (by the context-string embedding) records an observation (similarity, judge
    verdict); a request whose nearest entry is wrong, or that finds no entry, is inserted."""

    def __init__(self, w, E, k, capacity):
        self.w, self.E = w, E
        self.k = None if k == ALL else int(k)
        cap = capacity if self.k is None else self.k
        self.ent = np.full(cap, -1, dtype=np.int64)
        self.emb = np.zeros((cap, E.shape[1]), dtype=np.float32)
        self.conv = np.full(cap, -1, dtype=np.int64)
        self.last = np.zeros(cap, dtype=np.int64)
        self.obs = [None] * cap
        self.n = self.clock = 0

    def nearest(self, idx):
        if self.n == 0:
            return np.full(len(idx), -1), np.zeros(len(idx))
        S = self.E[idx] @ self.emb[: self.n].T
        S[self.w.conv_code[idx][:, None] == self.conv[None, : self.n]] = -np.inf
        j = np.argmax(S, axis=1)
        s = S[np.arange(len(idx)), j]
        ok = np.isfinite(s)
        return np.where(ok, j, -1), np.where(ok, np.round(s.astype(float), 3), 0.0)

    def insert(self, t):
        self.clock += 1
        if self.k is not None and self.n >= self.k:
            j = int(np.argmin(self.last[: self.n]))       # evict the least recently used entry
        else:
            j = self.n
            self.n += 1
        self.ent[j], self.emb[j], self.conv[j], self.last[j] = int(t), self.E[t], self.w.conv_code[t], self.clock
        self.obs[j] = list(_VC_PSEUDO)

    def replay(self, order):
        """Replay a log, VC_BATCH requests per labelling round (vCache labels asynchronously)."""
        for s0 in range(0, len(order), VC_BATCH):
            tb = order[s0:s0 + VC_BATCH]
            J, S = self.nearest(tb)
            ENT = np.where(J >= 0, self.ent[np.maximum(J, 0)], -1)
            for j in J:
                self.clock += 1
                if j >= 0:
                    self.last[j] = self.clock
            for i, t in enumerate(tb):
                j = J[i]
                if j < 0:
                    self.insert(t)
                    continue
                if self.ent[j] != ENT[i]:                   # evicted during this round: the update is dropped
                    continue
                if S[i] < VC_FLOOR:
                    y = 0
                else:
                    a = self.w.labels.lookup([t], [ENT[i]])[0]
                    if np.isnan(a):
                        continue
                    y = int(a)
                self.obs[j].append((float(S[i]), y))
                if y == 0:
                    self.insert(t)

    def frozen(self, idx):
        """Nearest entry, and the correctness bound α (NaN: the entry still explores) of every request."""
        J, S = np.empty(len(idx), dtype=int), np.empty(len(idx))
        for s0 in range(0, len(idx), 1024):
            J[s0:s0 + 1024], S[s0:s0 + 1024] = self.nearest(idx[s0:s0 + 1024])
        alpha, fits = np.full(len(idx), np.nan), {}
        for i in np.flatnonzero(J >= 0):
            o = self.obs[J[i]]
            if len(o) < _VC_MIN_OBS:
                continue
            if J[i] not in fits:
                fits[J[i]] = vc_fit(o)
            if fits[J[i]] is not None:
                alpha[i] = vc_alpha(fits[J[i]], S[i])
        return np.where(J >= 0, self.ent[np.maximum(J, 0)], -1), alpha, int(self.n)


def replay_order(w, idx, seed):
    """Conversations in a seeded random order, the turns of each conversation in order."""
    idx = np.asarray(idx, dtype=int)
    convs = np.unique(w.conv_code[idx])
    rank = np.zeros(int(w.conv_code.max()) + 1, dtype=np.int64)
    rank[convs] = np.random.default_rng(seed).permutation(len(convs))
    return idx[np.lexsort((w.pos_in_conv[idx], rank[w.conv_code[idx]]))]


# ---------------------------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------------------------

METHODS = ["CUCB-SC-MT", "ElastiCache-MT", "ContextCache", "vCache", "Joint-state threshold", "Single-turn"]


def run(w, cost, risk, methods=METHODS, log=print):
    """Select every method's setting on validation at each budget and evaluate it on test.
    Returns a table with one row per (method, budget) and the per-request test losses."""
    w.query_prefix = np.vstack([np.zeros((1, w.E_query.shape[1])),
                                np.cumsum(w.E_query[w.sorted_order].astype(np.float64), axis=0)])
    val_total = float(w.cost[w.val].sum())
    candidates = []   # (method, budget, [(setting, validation Decisions)], test function)

    # CUCB-SC-MT
    if "CUCB-SC-MT" in methods:
        eps_grid = epsilon_grid(w)
        log(f"  CUCB-SC-MT: ε grid {eps_grid}")
        tune = {e: CUCBSCMT(w, cost, risk, e, "tune") for e in eps_grid}
        final = {}
        def final_phase(e):
            if e not in final:
                final[e] = CUCBSCMT(w, cost, risk, e, "final")
            return final[e]
        for k in BUDGETS + [ALL]:
            eps_k = [e for e in eps_grid if tune[e].usable(k) and (k == ALL or final_phase(e).usable(k))]
            sweep = [((e, g), tune[e].decisions(k, g)) for e in eps_k for g in GAMMAS]
            candidates.append(("CUCB-SC-MT", k, sweep, lambda s, k=k: final_phase(s[0]).decisions(k, s[1]),
                               lambda s: f"ε = {s[0]:g}, γ̂ = {s[1]:g}"))

    # ElastiCache-MT: top-1 cosine of the context-string embedding
    if "ElastiCache-MT" in methods:
        E = w.E_context
        order_cal, order_src = lfu_order(E, w.conv_ids, w.cal), lfu_order(E, w.conv_ids, w.src)
        for k in BUDGETS + [ALL]:
            caches = (w.cal, w.src) if k == ALL else (lfu_cache(order_cal, w.cal, k), lfu_cache(order_src, w.src, k))
            cv, sv = top1(E, caches[0], w.val)
            ct, st = top1(E, caches[1], w.test)
            grid = EC_GRID if k == ALL else EC_GRID[EC_GRID >= EC_FLOOR - 1e-9]
            sweep = [(t, Decisions(w, w.val, cv, sv >= t)) for t in grid]
            candidates.append(("ElastiCache-MT", k, sweep,
                               lambda t, ct=ct, st=st, n=len(caches[1]): Decisions(w, w.test, ct, st >= t, n),
                               lambda t: f"cos ≥ {t:.3f}"))

    # ContextCache
    if "ContextCache" in methods:
        order_cal, order_src = lfu_order(w.E_query, w.conv_ids, w.cal), lfu_order(w.E_query, w.conv_ids, w.src)
        for k in BUDGETS + [ALL]:
            caches = (w.cal, w.src) if k == ALL else (lfu_cache(order_cal, w.cal, k), lfu_cache(order_src, w.src, k))
            v = context_cache_retrieve(w, caches[0], w.val)
            t_ = context_cache_retrieve(w, caches[1], w.test)
            grid = CC_GRID if k == ALL else CC_GRID[CC_GRID >= CC_FLOOR - 1e-9]
            sweep = [((tq, tc), Decisions(w, w.val, v[0], context_cache_reuse(v[1], v[2], v[3], tq, tc)))
                     for tq in grid for tc in grid]
            candidates.append(("ContextCache", k, sweep,
                               lambda s, t_=t_, n=len(caches[1]): Decisions(
                                   w, w.test, t_[0], context_cache_reuse(t_[1], t_[2], t_[3], *s), n),
                               lambda s: f"q ≥ {s[0]:.3f}, ctx ≥ {s[1]:.3f}"))

    # vCache: replay the calibration log (cache for validation), then the validation log (cache for test)
    if "vCache" in methods:
        U = np.random.default_rng(SEED + 51).random(w.n)
        order_cal, order_val = replay_order(w, w.cal, SEED + 51), replay_order(w, w.val, SEED + 52)
        for k in BUDGETS + [ALL]:
            vc = VCache(w, w.E_context, k, len(order_cal) + len(order_val))
            vc.replay(order_cal)
            cv, av, nv = vc.frozen(w.val)
            vc.replay(order_val)
            ct, at, nt = vc.frozen(w.test)
            sweep = [(d, Decisions(w, w.val, cv, np.isfinite(av) & (U[w.val] > vc_tau(np.nan_to_num(av), d)), nv))
                     for d in VC_DELTAS]
            candidates.append(("vCache", k, sweep,
                               lambda d, ct=ct, at=at, nt=nt: Decisions(
                                   w, w.test, ct, np.isfinite(at) & (U[w.test] > vc_tau(np.nan_to_num(at), d)), nt),
                               lambda d: f"δ = {d:g}"))

    # Joint-state threshold: ElastiCache-MT's rule on CUCB-SC-MT's state x
    if "Joint-state threshold" in methods:
        order_cal, order_src = lfu_order(w.X, w.conv_ids, w.cal), lfu_order(w.X, w.conv_ids, w.src)
        for k in BUDGETS + [ALL]:
            caches = (w.cal, w.src) if k == ALL else (lfu_cache(order_cal, w.cal, k), lfu_cache(order_src, w.src, k))
            cv, sv = top1(w.X, caches[0], w.val)
            ct, st = top1(w.X, caches[1], w.test)
            sweep = [(t, Decisions(w, w.val, cv, (sv >= JS_FLOOR - 1e-9) & (sv >= t))) for t in JS_GRID]
            candidates.append(("Joint-state threshold", k, sweep,
                               lambda t, ct=ct, st=st, n=len(caches[1]): Decisions(w, w.test, ct, st >= t, n),
                               lambda t: f"cos ≥ {t:.3f}"))

    # Single-turn CUCB-SC-Cont
    if "Single-turn" in methods:
        st = SingleTurn(w)
        for k in BUDGETS + [ALL]:
            sweep = [((e, g), st.decisions(e, k, g, "tune")) for e in ST_EPS for g in ST_GAMMAS]
            candidates.append(("Single-turn", k, sweep, lambda s, k=k: st.decisions(s[0], k, s[1], "final"),
                               lambda s: f"ε = {s[0]:g}, γ̂ = {s[1]:g}"))

    boot = Bootstrap(w)
    rows, losses = [], {}
    for method, k, sweep, test_fn, fmt in candidates:
        setting, val_loss, judged = select(sweep, val_total)
        if not judged:
            log(f"  note: every {method} setting at k = {k} has more than 1% unjudged validation hits")
        d = test_fn(setting)
        loss = d.losses()
        losses[(method, k)] = loss
        point, lo, hi = boot(loss)
        s = d.summary()
        rows.append({"method": method, "budget": k, "setting": fmt(setting),
                     "params": json.dumps([round(float(x), 6) for x in np.atleast_1d(setting)]), "entries": d.entries,
                     "val_loss": 100 * val_loss * len(w.val) / val_total, "loss": point, "loss_lo": lo,
                     "loss_hi": hi, "hits": s["hits"], "rejected": s["rejected"], "unjudged": s["unjudged"]})
        log(f"  {method:22s} {str(k):>4s}  {fmt(setting):24s} test loss {point:5.1f} [{lo:.1f}, {hi:.1f}]"
            + (f"  ({s['unjudged']} unjudged hits)" if s["unjudged"] else ""))
    return pd.DataFrame(rows), losses
