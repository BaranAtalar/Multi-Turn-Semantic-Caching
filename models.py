"""Kernel models of the generation cost and of the reuse risk (Nyström kernel ridge regression).

* Cost: c(z) = c_in(z) + c_out(z), where c_in (the input tokens) is known before the LLM call and
  c_out (the output tokens) is learned. The pessimistic cost is c̄(z) = c_in + ĉ_out + β_c σ_c.
* Risk: r(z, u) is the probability that the cached response of state u is an unacceptable answer
  for request z. It is learned from judge-labeled pairs with features ψ(z, u). The pessimistic
  risk is r̂ + β_r σ_r.
"""
import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, cholesky, solve_triangular
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from data import SEED

DELTA = 0.05                                      # confidence level of β_c and β_r
COST_RANK, COST_SCALES, COST_RIDGES = 1024, [0.5, 1.0, 2.0], [1e-3, 1e-2, 1e-1]
RISK_RANK, RISK_SCALES, RISK_RIDGES = 512, [0.125, 0.25, 0.5, 1.0, 2.0], [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
RISK_FOLDS, RISK_BINS = 5, 20
PCA_DIM, PCA_WEIGHT = 8, 0.5                      # ψ: 8 principal components per embedding block, weight 1/2


class NystromKRR:
    """RBF kernel ridge regression on Nyström features, with a posterior width σ(x).

    Landmarks are distinct rows; near-zero eigenvalues of the landmark kernel are dropped, so
    ||z(x)||^2 <= 1 holds and σ(x) cannot collapse to 0 through round-off.
    """

    def __init__(self, rank, ridge, gamma, seed=SEED, center=True, eig_tol=1e-8):
        self.rank, self.ridge, self.gamma, self.seed = int(rank), float(ridge), float(gamma), int(seed)
        self.center, self.eig_tol = center, eig_tol

    @staticmethod
    def rbf(A, B, gamma):
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        sq = np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :] - 2.0 * (A @ B.T)
        return np.exp(-gamma * np.maximum(sq, 0.0))

    def features(self, X, batch=4096):
        if not len(X):
            return np.zeros((0, self.proj.shape[1]))
        return np.vstack([self.rbf(X[s:s + batch], self.landmarks, self.gamma) @ self.proj
                          for s in range(0, len(X), batch)])

    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float64)
        rows = X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
        _, distinct = np.unique(rows, return_index=True)
        pick = np.random.default_rng(self.seed).permutation(distinct)[: self.rank]
        self.landmarks = X[np.sort(pick)].astype(np.float64)
        S, U = np.linalg.eigh(self.rbf(self.landmarks, self.landmarks, self.gamma))
        keep = S > self.eig_tol * S.max()
        self.proj = U[:, keep] / np.sqrt(S[keep])
        Z = self.features(X)
        self.y_mean = float(y.mean()) if self.center else 0.0
        self.chol = cholesky(Z.T @ Z + self.ridge * np.eye(Z.shape[1]), lower=True, check_finite=False)
        self.theta = cho_solve((self.chol, True), Z.T @ (y - self.y_mean), check_finite=False)
        return self

    def predict(self, X, batch=8192):
        """Posterior mean and width at the rows of X."""
        X = np.asarray(X, dtype=np.float32)
        means, stds = [], []
        for s in range(0, len(X), batch):
            Z = self.features(X[s:s + batch])
            means.append(Z @ self.theta + self.y_mean)
            V = solve_triangular(self.chol, Z.T, lower=True, check_finite=False)
            gap = np.maximum(1.0 - np.sum(Z * Z, axis=1), 0.0)          # Nyström approximation gap
            stds.append(np.sqrt(gap + self.ridge * np.sum(V * V, axis=0)))
        if not means:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(means), np.concatenate(stds)


def calibrate_beta(y, mean, std, delta=DELTA):
    """Smallest β with y <= mean + β std for a 1 - δ fraction of held-out points."""
    ratio = (np.asarray(y, float) - np.asarray(mean, float)) / np.maximum(np.asarray(std, float), 1e-6)
    return float(max(0.0, np.quantile(ratio, 1.0 - delta)))


def median_sq_distance(X, seed=SEED, n_pairs=30000):
    """Median heuristic for the RBF bandwidth, over random pairs of rows."""
    X = np.asarray(X, dtype=np.float32)
    r = np.random.default_rng(seed)
    a, b = r.integers(0, len(X), size=n_pairs), r.integers(0, len(X), size=n_pairs)
    sq = np.sum((X[a] - X[b]) ** 2, axis=1)
    sq = sq[sq > 1e-12]
    return float(np.median(sq)) if len(sq) else 1.0


# ---------------------------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------------------------

class CostModel:
    """Output-cost KRR on the state x. Its bandwidth and ridge minimize the validation error of a
    model fit on calibration states; β_c covers the validation residuals with probability 1 - δ."""

    def __init__(self, w):
        self.w = w
        self.gamma0 = 1.0 / median_sq_distance(w.X[w.cal])
        best = None
        for scale in COST_SCALES:
            for ridge in COST_RIDGES:
                m = NystromKRR(COST_RANK, ridge, scale * self.gamma0).fit(w.X[w.cal], w.cost_out[w.cal])
                err = np.mean(np.abs(np.clip(w.cost_in[w.val] + m.predict(w.X[w.val])[0], 0, 1) - w.cost[w.val]))
                if best is None or err < best[0]:
                    best = (err, scale, ridge, m)
        _, self.scale, self.ridge, self.tune_model = best
        mu, sd = self.tune_model.predict(w.X[w.val])
        self.beta = calibrate_beta(w.cost_out[w.val], mu, sd)
        self.final_model = NystromKRR(COST_RANK, self.ridge, self.scale * self.gamma0).fit(
            w.X[w.src], w.cost_out[w.src])

    def ucb(self, idx, phase):
        """c̄(z) = c_in(z) + ĉ_out(z) + β_c σ_c(z), clipped to [0, 1]. phase 'tune' uses the model fit
        on calibration states (for validation), 'final' the one fit on all source states (for test)."""
        model = self.tune_model if phase == "tune" else self.final_model
        mu, sd = model.predict(self.w.X[idx])
        return np.clip(self.w.cost_in[idx] + mu + self.beta * sd, 0.0, 1.0)


# ---------------------------------------------------------------------------------------------
# Risk model
# ---------------------------------------------------------------------------------------------

class PairFeatures:
    """ψ(z, u): similarities of the two states' request, history and state embeddings, of z's request
    and u's answer, exact-match flags, turn positions and history lengths (16 scalars), and the
    leading principal components of x_z, x_u and u's answer embedding (8 each).

    The 'same intent' and 'same task' flags come from an LLM enrichment of each request that is not
    used here, so they are 0; they are kept so that ψ has the same 40 dimensions as in the paper.
    """

    def __init__(self, w):
        self.w = w
        self.p_state = PCA(n_components=PCA_DIM, random_state=SEED).fit(w.X[w.cal]).transform(w.X).astype(np.float32)
        self.p_answer = PCA(n_components=PCA_DIM, random_state=SEED).fit(
            w.E_answer[w.cal]).transform(w.E_answer).astype(np.float32)
        self.has_hist = (w.history_tokens > 0).astype(np.float32)
        self.log_turn = np.log1p(w.pos_in_conv).astype(np.float32)
        self.log_hist = np.log1p(w.history_tokens).astype(np.float32)

    @staticmethod
    def rowdot(A, B, t, c, chunk=8192):
        out = np.empty(len(t), dtype=np.float32)
        for s in range(0, len(t), chunk):
            out[s:s + chunk] = np.einsum("ij,ij->i", A[t[s:s + chunk]], B[c[s:s + chunk]])
        return out

    def raw(self, t, c):
        w = self.w
        t, c = np.asarray(t, dtype=int), np.asarray(c, dtype=int)
        zero = np.zeros(len(t), dtype=np.float32)
        scalars = np.stack([
            self.rowdot(w.X, w.X, t, c),
            self.rowdot(w.E_query, w.E_query, t, c),            # request vs request (the "rewrite" block)
            self.rowdot(w.E_history, w.E_history, t, c),
            self.rowdot(w.E_query, w.E_query, t, c),
            self.rowdot(w.E_query, w.E_answer, t, c),           # z's request vs u's cached answer
            zero, zero,                                         # same intent, same task (not used)
            (w.query_code[t] == w.query_code[c]).astype(np.float32),
            (w.query_lower_code[t] == w.query_lower_code[c]).astype(np.float32),
            self.has_hist[t], self.has_hist[c],
            self.log_turn[t], self.log_turn[c], np.abs(self.log_turn[t] - self.log_turn[c]),
            self.log_hist[t], self.log_hist[c],
        ], axis=1).astype(np.float32)
        return np.concatenate([scalars, self.p_state[t], self.p_state[c], self.p_answer[c]], axis=1)


class RiskModel:
    """Reuse-risk KRR on standardized pair features. Bandwidth and ridge minimize the out-of-fold
    Brier score (folds grouped by conversation). β_r makes r̂ + β_r σ_r cover the binned empirical
    risk of 1 - δ of the out-of-fold pairs."""

    def __init__(self, w, features):
        self.features = features
        pairs = w.risk_pairs
        t = pairs["target_state_id"].map(w.position).to_numpy()
        c = pairs["candidate_state_id"].map(w.position).to_numpy()
        y = 1.0 - w.labels.lookup(t, c)                          # 1 = unacceptable reuse
        F = features.raw(t, c)
        self.mu, self.sd = F.mean(axis=0), np.maximum(F.std(axis=0), 1e-6)
        self.weights = np.ones(F.shape[1], dtype=np.float32)
        self.weights[16:] = PCA_WEIGHT
        F = self.standardize(F)
        self.F_train = F                                         # standardized training features
        gamma0 = 1.0 / median_sq_distance(F)
        groups = w.conv_code[t]
        folds = list(GroupKFold(n_splits=min(RISK_FOLDS, len(np.unique(groups)))).split(F, y, groups))
        best = None
        for scale in RISK_SCALES:
            for ridge in RISK_RIDGES:
                p, s = np.zeros(len(y)), np.zeros(len(y))
                for tr, te in folds:
                    p[te], s[te] = NystromKRR(RISK_RANK, ridge, scale * gamma0).fit(F[tr], y[tr]).predict(F[te])
                p = np.clip(p, 0, 1)
                brier = float(np.mean((p - y) ** 2))
                if best is None or brier < best[0]:
                    best = (brier, scale, ridge, p, s)
        _, self.scale, self.ridge, p, s = best
        self.beta = calibrate_beta(binned_rate(y, p, RISK_BINS), p, s)
        self.gamma = self.scale * gamma0
        self.model = NystromKRR(RISK_RANK, self.ridge, self.gamma).fit(F, y)

    def standardize(self, F):
        return ((F - self.mu) / self.sd * self.weights).astype(np.float32)

    def predict(self, t, c):
        """r̂(z_t, u_c) clipped to [0, 1] and its width σ_r."""
        mu, sd = self.model.predict(self.standardize(self.features.raw(t, c)))
        return np.clip(mu, 0.0, 1.0).astype(np.float32), sd.astype(np.float32)

    def matrix(self, rows, cols, batch=50_000):
        """r̂ and σ_r for every (row state, column state) pair."""
        rows, cols = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
        n, m = len(rows), len(cols)
        MU, SD = np.empty((n, m), dtype=np.float32), np.empty((n, m), dtype=np.float32)
        per = max(1, batch // m)
        for s in range(0, n, per):
            ii = np.arange(s, min(n, s + per))
            mu, sd = self.predict(np.repeat(rows[ii], m), np.tile(cols, len(ii)))
            MU[ii], SD[ii] = mu.reshape(len(ii), m), sd.reshape(len(ii), m)
        return MU, SD


def binned_rate(y, p, n_bins):
    """Empirical rate of y within quantile bins of p (an estimate of the conditional risk)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    return pd.Series(y).groupby(b).transform("mean").to_numpy()
